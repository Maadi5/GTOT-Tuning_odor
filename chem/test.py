import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
from splitters import predetermined_split
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from pathlib import Path
from model import GNN_graphpred
from ftlib.finetune.delta import IntermediateLayerGetter, L2Regularization, get_attribute
from ftlib.finetune.delta import SPRegularization, FrobeniusRegularization
import json
from commom.meter import AverageMeter, ProgressMeter
from commom.eval import Meter
from tqdm import tqdm
from loader import MoleculeDataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Assuming you have extracted embeddings in the 'embeddings' variable
# Create a StandardScaler instance
scaler = StandardScaler()



path_to_dataset = os.path.join(os.getcwd(), 'dataset_smiles.json')

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def _load_odour_dataset_json(dataset_path, split_json_path, split= 'test'):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(dataset_path, index_col=False)

    jsonfile = json.load(open(split_json_path, encoding='utf8'))
    smiles_list = jsonfile[split]
    #smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = list(input_df.columns[:-2])
    print('tasks: ', tasks)
    labels = input_df[tasks]
    # convert 0 to 1
    labels = labels.replace(0, 1)
    # convert nan to 0
    labels = labels.fillna(-1)
    for idx, row in labels.iterrows():
        row_values = [v for k, v in dict(row).items()]
        #print('sum of row values in dataset creator: ', sum(row_values))
        if sum(row_values) == 0:
            print("Missing labels for sample")
    for ar in labels.values:
        #print(ar)
        if sum(ar) == 0.0:
            print('incomplete labels')
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def Inference(args, model, device, loader, source_getter, target_getter, plot_confusion_mat=False):
    model.eval()

    loss_sum = []
    eval_meter = Meter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)

        with torch.no_grad():

            intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # Fit the scaler on the embeddings and transform the data
            standardized_embeddings = scaler.fit_transform(output_t)
            pred = output_t
            print('prediction shape: ', pred.shape)
            n_components = 2  # You can choose the number of components you want to analyze
            pca = PCA(n_components=n_components)

            # Fit the PCA model on your standardized embeddings
            pca.fit(standardized_embeddings)

            # Transform the data to the new reduced-dimensional space
            reduced_embeddings = pca.transform(standardized_embeddings)
            print('reduced embeddings: ', reduced_embeddings.shape)
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
            plt.title('2D Visualization of Model Activations')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()

        y = batch.y.view(pred.shape)
        eval_meter.update(pred, y, mask=y ** 2 > 0)

    metric = np.mean(eval_meter.compute_metric('roc_auc_score_finetune'))

    return metric, sum(loss_sum)

def test(args, split_json_path):
    num_tasks = 133
    device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)
    args.device = device

    if args.gnn_type == 'gin':
        return_layers = ['gnn.gnns.4.mlp.2']
    elif args.gnn_type == 'gcn':
        return_layers = ['gnn.gnns.4.linear']
    elif args.gnn_type in ['gat', 'gat_ot']:
        return_layers = [
            'gnn.gnns.4.weight_linear']

    target_json = json.load(open(split_json_path, encoding='utf8'))

    smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
    dataset = MoleculeDataset(os.path.join(args.data_path, args.dataset), dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset,smiles_distrib = predetermined_split(dataset= dataset, target_json= target_json, smiles_list= smiles_list)

    device = args.device
    ## finetuned model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, args=args)
    ## pretrained model
    source_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                                 graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, args=args)
    # get the output feature map of the mediate layer in full model
    source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
    target_getter = IntermediateLayerGetter(model, return_layers=return_layers)


    if not args.input_model_file in ["", 'none']:
        print('loading pretrain model from', args.input_model_file)
        source_model.from_pretrained(args.input_model_file)
    model.load_state_dict(torch.load(os.path.join(args.data_path, args.dataset, 'final_test_weights.pt'))['model_state_dict'])
    model.to(device)
    backbone = model.gnn
    print("backbone: ", backbone)
    classifier = model
    print("classifier: ", classifier)
    print('test dataset: ', test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_acc, test_loss = Inference(args, model, device, test_loader, source_getter, target_getter,
                                    plot_confusion_mat=True)

    print('test accuracy: ', test_acc)
    print('test loss: ', test_loss)

if __name__ == "__main__":
    from parser import *

    args = get_parser()
    print(args)
    elapsed_times = []
    seed_nums = list(range(10))
    test(args, split_json_path='dataset_smiles.json')