import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from pathlib import Path
from model import GNN_graphpred
import json



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
    print('tasks: '. tasks)
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



def test(args, input_df_path, split_json_path):
    device = args.device
    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
        model.to(device)
        backbone = model.gnn
    classifier = model

    smiles_list, rdkit_mol_objs, label_values = \
        _load_odour_dataset_json(dataset_path= input_df_path, split_json_path= split_json_path)

    data_list = []
    for i in range(len(smiles_list)):
        # print(i)
        rdkit_mol = rdkit_mol_objs[i]
        # # convert aromatic bonds to double bonds
        # Chem.SanitizeMol(rdkit_mol,
        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        data = mol_to_graph_data_obj_simple(rdkit_mol)
        # manually add mol id
        data.id = torch.tensor(
            [i])  # id here is the index of the mol in
        # # the dataset
        # data.y = torch.tensor(labels[i, :])
        # print('sum of data: ', torch.sum(data.y))
        if torch.sum(data.y) == 0:
            print('sum 0 data: ', data.id)
        data_list.append(data)
        # data_smiles_list.append(smiles_list[i])


if __name__ == "__main__":
    from parser import *

    args = get_parser()
    print(args)
    elapsed_times = []
    seed_nums = list(range(10))
