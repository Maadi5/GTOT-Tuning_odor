from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix #, plot_confusion_matrix
from torchmetrics.functional import precision_recall
from sklearn.preprocessing import MultiLabelBinarizer
import math
import torch.nn.functional as F
import torch.optim as optim
import copy
import json

from tqdm import tqdm
import numpy as np

from model import GNN_graphpred

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

# sys.path.append('..')
from commom.meter import AverageMeter, ProgressMeter
from commom.eval import Meter
import os, random
import shutil
from commom.early_stop import EarlyStopping
from commom.run_time import Runtime

import torch
from torch import nn
from ftlib.finetune.stochnorm import convert_model
from ftlib.finetune.bss import BatchSpectralShrinkage
from ftlib.finetune.delta import IntermediateLayerGetter, L2Regularization, get_attribute
from ftlib.finetune.delta import SPRegularization, FrobeniusRegularization

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorboard.plugins import pr_curve

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_labels = 133

# Define the metrics
precision = tf.metrics.Precision()
recall = tf.metrics.Recall()

# Define the Tensorboard summary writers
train_summary_writer = tf.summary.create_file_writer('./logs/train')
val_summary_writer = tf.summary.create_file_writer('./logs/val')

def get_confusion_matrix(y_true, y_pred, num_classes):
    # Compute the confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)

    # Normalize the confusion matrix
    cm = tf.cast(cm, tf.float32)
    row_sums = tf.reduce_sum(cm, axis=1)
    cm_norm = cm / row_sums[:, tf.newaxis]

    return cm_norm


def plot_confusion_matrix(y_true, y_pred, class_names):
    num_classes = len(class_names)

    # Convert one-hot encoded predictions to label indices
    y_pred_labels = tf.argmax(y_pred, axis=-1)

    # Compute confusion matrix
    cm = tf.math.confusion_matrix(y_true, y_pred_labels, num_classes=num_classes)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel="True label",
           xlabel="Predicted label")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Add the confusion matrix as an image summary to TensorBoard
    summary = tf.Summary()
    summary_image = tf.Summary.Image(
        tensor=tf.expand_dims(cm, axis=-1).astype(np.float32),
        colormap=11)  # 11 corresponds to the 'Blues' color map
    summary.value.add(tag='Confusion_Matrix', image=summary_image)
    writer.add_summary(summary)
    writer.flush()
# Define a function to log the confusion matrix to TensorBoard
def log_confusion_matrix(writer, cm, step):
    # Create a figure and plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', ax=ax)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Convert the plot to an image and log it to TensorBoard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary = tf.summary.image('Confusion Matrix', image, step=step)
    writer.add_summary(summary)
    buf.close()
    plt.close()

logdir = "logs/"
writer = tf.summary.create_file_writer(logdir)

def get_metrics_2(y_true, y_pred):
    # Compute confusion matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)
    cm = np.array([c[1] for c in cm])

    # Compute precision and recall
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Confusion Matrix')
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)

    # Print metrics
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    plt.show()



criterion = nn.BCEWithLogitsLoss(reduction="none")
import warnings
warnings.filterwarnings('ignore')

def get_metrics(y_true, y_pred):
    # Compute confusion matrix
    cm = multilabel_confusion_matrix(y_true, y_pred)

    # Compute precision and recall
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')

    # Print metrics
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print("Confusion Matrix:")
    for i, cm_i in enumerate(cm):
        print(f"Label {i}:")
        print(cm_i)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    ## cuda
    # torch.cuda.current_device()
    torch.cuda._initialized = True

def calculate_channel_attention(dataset, return_layers, args):
    device = args.device
    train_meter = Meter()

    model = GNN_graphpred(args.num_layer, args.emb_dim, args.num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type)

    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
        model.to(device)
        backbone = model.gnn
    classifier = model
    data_loader = DataLoader(dataset, batch_size=args.attention_batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=False)

    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(
        math.log(0.1) / args.attention_lr_decay_epochs))

    # to save the change of loss when the output channel(every col in W) weights are masked.
    channel_weights = []
    for layer_id, name in enumerate(return_layers):
        layer = get_attribute(classifier, name)
        layer_channel_weight = [0] * layer.out_features
        channel_weights.append(layer_channel_weight)

    # train the classifier
    classifier.train()
    classifier.gnn.requires_grad = False

    print("## Pretrain a classifier to calculate channel attention.")
    for epoch in range(args.attention_epochs):
        losses = AverageMeter('Loss', ':3.2f')
        cls_accs = AverageMeter('roc_auc_socre', ':3.1f')
        progress = ProgressMeter(
            len(data_loader),
            [losses, cls_accs],
            prefix="Epoch: [{}]".format(epoch))

        # for i, data in enumerate(data_loader):
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # loss = criterion(outputs, labels)

            y = batch.y.view(pred.shape).to(torch.float64)

            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            loss = torch.sum(loss_mat) / torch.sum(is_valid)

            train_meter.update((pred + 1) / 2, y, mask=is_valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cls_acc = accuracy(outputs, labels)[0]
            cls_acc = np.mean(train_meter.compute_metric('roc_auc_score_finetune'))
            losses.update(loss.item(), batch.batch.size(0))
            cls_accs.update(cls_acc.item(), batch.batch.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
        lr_scheduler.step()

    # calculate the channel attention
    print('Calculating channel attention.')
    classifier.eval()
    if args.attention_iteration_limit > 0:
        total_iteration = min(len(data_loader), args.attention_iteration_limit)
    else:
        total_iteration = len(args.data_loader)

    progress = ProgressMeter(
        total_iteration,
        [],
        prefix="Iteration: ")

    for i, batch in enumerate(data_loader):
        if i >= total_iteration:
            break
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat,
                               torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss_0 = torch.sum(loss_mat) / torch.sum(is_valid)

        if i % 20 == 0:
            progress.display(i)
        # for layer_id, name in enumerate(tqdm(return_layers)):
        for layer_id, name in enumerate(return_layers):
            layer = get_attribute(classifier, name)
            for j in range(layer.out_features):
                tmp = classifier.state_dict()[name + '.weight'][j,].clone()
                classifier.state_dict()[name + '.weight'][j,] = 0.0
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss_mat_1 = criterion(pred.double(), (y + 1.0) / 2)
                loss_mat_1 = torch.where(is_valid, loss_mat_1,
                                         torch.zeros(loss_mat_1.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss_1 = torch.sum(loss_mat_1) / torch.sum(is_valid)

                difference = loss_1 - loss_0
                difference = difference.detach().cpu().numpy().item()
                history_value = channel_weights[layer_id][j]
                # 计算 mean vlaue of the increasement of loss
                channel_weights[layer_id][j] = 1.0 * (i * history_value + difference) / (i + 1)
                # recover the weight of model
                classifier.state_dict()[name + '.weight'][j,] = tmp

    channel_attention = []
    for weight in channel_weights:
        weight = np.array(weight)
        weight = (weight - np.mean(weight)) / np.std(weight)
        weight = torch.from_numpy(weight).float().to(device)
        channel_attention.append(F.softmax(weight / 5, dim=-1).detach())
    return channel_attention


def train_epoch(args, model, device, loader, optimizer, weights_regularization, backbone_regularization,
                head_regularization, target_getter,
                source_getter, bss_regularization, scheduler, epoch, targets):
    model.train()

    meter = Meter()
    loss_epoch = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)
        if args.finetune_type == 'none':
            fea = fea_s = None
            intermediate_output_s, output_s = source_getter(batch.x, batch.edge_index, batch.edge_attr,
                                                            batch.batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
            intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = output_t
        else:

            intermediate_output_s, output_s = source_getter(batch.x, batch.edge_index, batch.edge_attr,
                                                            batch.batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
            intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = output_t
            fea_s = source_getter._model.get_bottleneck()
            fea = target_getter._model.get_bottleneck()


            # intermediate_output_s
        y = batch.y.view(pred.shape).to(torch.float64)


        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        if args.debug:
            loss_mat = criterion(pred.double(), (y + 1) / 2)

        else:
            loss_mat = criterion(pred.double(), (y + 1) / 2)

        # loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)
        meter.update(pred, y, mask=is_valid)

        loss_reg_head = head_regularization()
        loss_reg_backbone = 0.0
        print_str = ''
        loss = torch.tensor([0.0], device=device)
        loss_bss = 0.0
        loss_weights = torch.tensor([0.0]).to(cls_loss.device)
        if args.regularization_type == 'feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        elif args.regularization_type == 'attention_feature_map':
            loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t)
        elif args.regularization_type == 'l2_sp':
            loss_reg_backbone = backbone_regularization()
        elif args.regularization_type == 'bss':
            fea = fea if fea is not None else model.get_bottleneck()
            loss_bss = bss_regularization(fea)  # if fea is not None else 0.0
        elif args.regularization_type == 'none':
            loss_reg_backbone = 0.0
            # loss_reg_head = 0.0
            loss_bss = 0.0
        elif args.regularization_type in ['gtot_feature_map',]:
            if args.trade_off_backbone > 0.0:
                loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t, batch)
            if False and 'best_' in args.tag:
                loss_weights = weights_regularization()
                print_str += f'loss_weights:{loss_weights:.5f}'
        else:
            loss_reg_backbone = backbone_regularization()

        loss = loss + cls_loss + args.trade_off_backbone * loss_reg_backbone + args.trade_off_head * loss_reg_head + args.trade_off_bss * loss_bss
        loss = loss + 0.1 * loss_weights
        # if torch.isnan(cls_loss):  # or torch.isnan(loss_reg_backbone):
        #     print(pred, loss_reg_backbone)
        #     raise
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10)
        optimizer.step()

        loss_epoch.append(cls_loss.item())

        #print('prediction: ')
        all_o_preds = []
        for p in pred.tolist():
            plist = []
            for idx, po in enumerate(p):
                if po>0.5:
                    plist.append(targets[idx])
            all_o_preds.append(plist)

        # print('gt: ')
        all_o_gt = []
        for p in ((y + 1) / 2).tolist():
            plist = []
            for idx, po in enumerate(p):
                if po>0.5:
                    plist.append(targets[idx])
            all_o_gt.append(plist)

    avg_loss = sum(loss_epoch) / len(loss_epoch)

    if scheduler is not None: scheduler.step()
    print(f'{"vanilla model || " if fea is None and args.norm_type == "none" else ""} '
          f'cls_loss:{avg_loss:.5f}, loss_reg_backbone: {args.trade_off_backbone * loss_reg_backbone:.5f} loss_reg_head:'
          f' {args.trade_off_head * loss_reg_head:.5f} bss_los: {args.trade_off_bss * loss_bss:.5f} ' + print_str)
    try:
        print('num_oversmooth:', backbone_regularization.num_oversmooth, end=' || ')
        backbone_regularization.num_oversmooth = 0
    except:
        pass

    metric = np.mean(meter.compute_metric('roc_auc_score_finetune'))

    # pr_recall = precision_recall(preds=torch.tensor(pred_all), target=torch.tensor(labels_all), average='macro',
    #                              mdmc_average=None, ignore_index=None,num_classes=num_classes, threshold=0.5, top_k=None, multiclass=None)
    return metric, avg_loss, all_o_preds, all_o_gt


def eval(args, model, device, loader):
    model.eval()

    loss_sum = []
    eval_meter = Meter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)
        # print('BATCH.BATCH VALUE: ', batch.batch)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(pred.shape)
            eval_meter.update(pred, y, mask=y ** 2 > 0)

            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1.0) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss_sum.append(cls_loss.item())
            #print('prediction: ')
            all_o_preds = []
            for p in pred.tolist():
                plist = []
                for idx, po in enumerate(p):
                    if po>=0.5:
                        #plist.append(idx) #(targets[idx])
                        plist.append(1)
                    else:
                        plist.append(0)
                all_o_preds.append(plist)

            # print('gt: ')
            all_o_gt = []
            for p in ((y + 1) / 2).tolist():
                plist = []
                for idx, po in enumerate(p):
                    if po>=0.5:
                        #plist.append(idx) #(targets[idx])
                        plist.append(1)
                    else:
                        plist.append(0)
                all_o_gt.append(plist)


    metric = np.mean(eval_meter.compute_metric('roc_auc_score_finetune'))

    return metric, sum(loss_sum) / len(loss_sum), all_o_preds, all_o_gt


def Inference(args, model, device, loader, source_getter, target_getter, plot_confusion_mat=False):
    model.eval()

    loss_sum = []
    eval_meter = Meter()
    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=True)):
        batch = batch.to(device)
        with torch.no_grad():

            intermediate_output_t, output_t = target_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = output_t

        y = batch.y.view(pred.shape)
        eval_meter.update(pred, y, mask=y ** 2 > 0)

    metric = np.mean(eval_meter.compute_metric('roc_auc_score_finetune'))

    return metric, sum(loss_sum)



def main(args):
    device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)
    args.device = device


    print('Odour dataset name: ', args.dataset)
    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "odour":
        num_tasks = 133
    elif args.dataset == "odour_openpom":
        num_tasks = 138
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset == "zinc_standard_agent":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")
    args.num_tasks = num_tasks
    # set up dataset
    print(args.data_path, args.dataset)
    print('paths: ', os.path.join(args.data_path, args.dataset))
    dataset = MoleculeDataset(os.path.join(args.data_path, args.dataset), dataset=args.dataset)

    print(dataset)
    train_val_test = [0.8, 0.1, 0.1]
    smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()
    ds_df = pd.read_csv(os.path.join(args.data_path, args.dataset, 'raw', args.dataset + '.csv'), index_col=False)
    labels_all = list(ds_df.columns)[:-2]
    print('odors: ', labels_all)
    print('len labels: ', len(labels_all))
    if args.split == "scaffold":

        smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset, 'processed/smiles.csv'), header=None)[0].tolist()


        train_dataset, valid_dataset, test_dataset,smiles_distrib = scaffold_split(dataset, smiles_list, null_value=0,
                                                                    frac_train=train_val_test[0],
                                                                    frac_valid=train_val_test[1],
                                                                    frac_test=train_val_test[2],
                                                                    train_radio=args.train_radio,
                                                                    return_smiles=True,
                                                                    )

        print('test dataset: ', test_dataset)
        test_json = {'train': smiles_distrib[0], 'valid': smiles_distrib[1], 'test': smiles_distrib[2]}
        # json.dump(test_json, open(os.path.join(os.getcwd(), 'dataset_smiles.json'), 'w', encoding='utf8'), ensure_ascii=False)

        print(
            f"scaffold, train:test:val={len(train_dataset)}:{len(valid_dataset)}:{len(test_dataset)}, train_radio:{args.train_radio}")
        #print('train_0: ', train_dataset[0])
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=1.0, frac_valid=0.0,
                                                                  frac_test=0.0, seed=args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(os.path.join(args.data_path, args.dataset, 'processed/smiles.csv'), header=None)[
            0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0,
                                                                           frac_train=0.8, frac_valid=0.1,
                                                                           frac_test=0.1, seed=args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    # print('train_dataset_0: ', train_dataset[0])
    print('dataset_type: ', type(train_dataset))
    shuffle = True
    if args.debug:
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    # for b in train_loader:
        # print('train loader: ', b.x)
        # print(len(b.x))
        # print(len(b.y.tolist()))
        # print('sum_of_labels: ', torch.sum(b.y))
        # print('each y: ')
        # sumlist = []
        # for idx, val in enumerate(b.y.tolist()):
        #     sumlist.append(val)
        #     if idx % 32 == 0 and idx != 0:
        #         print('sum of label batch: ',sum(sumlist))
        #         sumlist = []

    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    ## finetuned model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, args=args)
    ## pretrained model
    source_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio,
                                 graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, args=args)

    if not args.input_model_file in ["", 'none']:
        print('loading pretrain model from', args.input_model_file)
        model.from_pretrained(args.input_model_file)
        source_model.from_pretrained(args.input_model_file)

    model.to(device)
    source_model.to(device)

    for param in source_model.parameters():
        param.requires_grad = False
    source_model.eval()

    ## one of baseline methods: StochNorm
    if args.norm_type == 'stochnorm':
        print('converting model with strochnorm')
        model = convert_model(model, p=args.prob)
        source_model = convert_model(source_model, p=args.prob)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # create intermediate layer getter
    if args.gnn_type == 'gin':

        return_layers = ['gnn.gnns.4.mlp.2']
    elif args.gnn_type == 'gcn':
        return_layers = ['gnn.gnns.4.linear']
    elif args.gnn_type in ['gat', 'gat_ot']:
        return_layers = [
            'gnn.gnns.4.weight_linear']
    else:
        raise NotImplementedError(args.gnn_type)

    # get the output feature map of the mediate layer in full model
    source_getter = IntermediateLayerGetter(source_model, return_layers=return_layers)
    target_getter = IntermediateLayerGetter(model, return_layers=return_layers)

    # get regularization for finetune
    weights_regularization = FrobeniusRegularization(source_model.gnn, model.gnn)
    backbone_regularization = lambda x: x
    bss_regularization = lambda x: x

    if args.regularization_type in ['gtot_feature_map']:
        ''' the proposed method GTOT-tuning'''
        from ftlib.finetune.gtot_tuning import GTOTRegularization
        backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
    #------------------------------ baselines --------------------------------------------
    elif args.regularization_type == 'l2_sp':
        backbone_regularization = SPRegularization(source_model.gnn, model.gnn)

    elif args.regularization_type == 'feature_map':
        from ftlib.finetune.delta import BehavioralRegularization
        backbone_regularization = BehavioralRegularization()

    elif args.regularization_type == 'attention_feature_map':
        from ftlib.finetune.delta import AttentionBehavioralRegularization
        attention_file = os.path.join('delta_attention', f'{args.gnn_type}_{args.dataset}_{args.attention_file}')
        if os.path.exists(attention_file):
            print("Loading channel attention from", attention_file)
            attention = torch.load(attention_file)
            attention = [a.to(device) for a in attention]
        else:
            print('attention_file', attention_file)
            attention = calculate_channel_attention(train_dataset, return_layers, args)
            torch.save(attention, attention_file)

        backbone_regularization = AttentionBehavioralRegularization(attention)



    elif args.regularization_type == 'bss':
        bss_regularization = BatchSpectralShrinkage(k=args.k)
        if args.debug:
            from ftlib.finetune.gtot_tuning import GTOTRegularization
            backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
    # ------------------------------ end --------------------------------------------
    elif args.regularization_type == 'none':
        backbone_regularization = lambda x: x
        bss_regularization = lambda x: x
        pass
    else:
        raise NotImplementedError(args.regularization_type)

    head_regularization = L2Regularization(nn.ModuleList([model.graph_pred_linear]))


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=6,
                                                           verbose=False,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-8,
                                                           eps=1e-08)
    save_model_name = os.path.join(args.save_path, f'{args.gnn_type}_{args.dataset}_{args.tag}.pt')
    stopper = EarlyStopping(mode='higher', patience=args.patience, filename=save_model_name)

    # if 0 and len(args.filename) != 0:
    if args.filename not in ['', "", 'none']:
        from tensorboardX import SummaryWriter

        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + f'/{args.dataset}/' + args.filename

        print('tensorboard file', fname)
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing tensorboard file.")
        else:
            print(f"Not existed file {fname}, create new one!")

        writer = SummaryWriter(fname)

    training_time = Runtime()
    test_time = Runtime()
    for epoch in range(1, args.epochs + 1):

        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
        training_time.epoch_start()
        train_acc, train_loss, all_o_preds, all_o_gt = train_epoch(args, model, device, train_loader, optimizer,
                                                             weights_regularization,
                                                             backbone_regularization,
                                                             head_regularization, target_getter,
                                                             source_getter, bss_regularization,
                                                             scheduler=None,
                                                             epoch=epoch, targets=smiles_list)

        training_time.epoch_end()

        print("====Evaluation")

        val_acc, val_loss, all_o_preds_val ,all_o_gt_val = eval(args, model, device, val_loader)
        test_time.epoch_start()
        test_acc, test_loss, all_o_preds_test ,all_o_gt_test = eval(args, model, device, test_loader)
        test_time.epoch_end()

        print('preds val: ', all_o_preds_test)
        print('gt val: ',all_o_gt_test)
        all_o_preds_test_bin = MultiLabelBinarizer().fit_transform(all_o_preds_test)
        all_o_gt_test_bin = MultiLabelBinarizer().fit_transform(all_o_gt_test)
        # print('preds_test: ')
        # print(all_o_preds_test)
        # print('gt_test: ')
        # print(all_o_gt_test)
        # confusion = confusion_matrix(all_o_gt_test, all_o_preds_test)
            # pr_recall = precision_recall(preds= torch.tensor(all_o_preds_test), target= torch.tensor(all_o_gt_test), average='macro', mdmc_average=None, ignore_index=None,
            #                             num_classes=133, threshold=0.5, top_k=None, multiclass=None)
        # get_metrics_2(y_true=all_o_gt_test, y_pred=all_o_preds_test)
        # print(confusion)
        #print(pr_recall)
        # Update the metrics
        #for i in range(len(all_o_preds_test)):
        # Check if y_true[i] or y_pred[i] has only one element
        # print('all_gt_i: ', all_o_gt_test[i])
        # if len(np.unique(np.argmax(all_o_gt_test[i], axis=-1))) == 1 or len(np.unique(np.argmax(all_o_preds_test[i], axis=-1))) == 1:
        #     continue
        print('COMING TO PREVISION RECALL CODE...')
        precision.update_state(all_o_gt_test, all_o_preds_test)
        recall.update_state(all_o_gt_test, all_o_preds_test)

        # Calculate the confusion matrix
        cm = confusion_matrix(np.argmax(all_o_gt_test, axis=-1), np.argmax(all_o_preds_test, axis=-1))

        # Log the metrics and confusion matrix to Tensorboard
        # Log the metrics and confusion matrix to Tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('precision', precision.result(), step=epoch)
            tf.summary.scalar('recall', recall.result(), step=epoch)
            tf.summary.text('confusion_matrix', np.array2string(cm), step=epoch)

        # Reset the metrics for the next batch
        precision.reset_states()
        recall.reset_states()

        try:
            scheduler.step(-val_acc)
        except:
            scheduler.step()

        if args.filename not in ['', "", 'none']:
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)
            writer.add_scalar('data/train loss', train_loss, epoch)
            writer.add_scalar('data/val loss', val_loss, epoch)
            writer.add_scalar('data/test loss', test_loss, epoch)
        args2 = copy.deepcopy(model.args)
        save_weights_path = os.path.join(os.getcwd(), args.dataset + '_weights' + str(epoch) + '.pt')
        torch.save({'model_state_dict': model.state_dict(), 'args': args2}, save_weights_path)
        if stopper.step(val_acc, model, test_score=test_acc, IsMaster=args.debug):
            print('stopper.step happening')
            stopper.report_final_results(i_epoch=epoch)
            break
        stopper.print_best_results(i_epoch=epoch, val_cls_loss=val_loss, train_acc=train_acc, val_score=val_acc,
                                   test_socre=test_acc, gnn_type=args.gnn_type,
                                   dataset=args.dataset, tag=args.tag)
    ## only inference, should set the --epoch 0
    inference = False
    if inference and args.debug:
        print('Inferencing')
        if stopper.best_model is None:
            print(stopper.filename)
            stopper.load_checkpoint(model)
            print('checkpint args', model.args)
        else:
            model.load_state_dict(stopper.best_model)
        model.to(device)
        test_acc, test_loss = Inference(args, model, device, test_loader, source_getter, target_getter,
                                        plot_confusion_mat=True)
        print(f'inference test_acc:{test_acc:.5f}')
        return test_acc, stopper.best_epoch, training_time

    training_time.print_mean_sum_time(prefix='Training')
    test_time.print_mean_sum_time(prefix='Test')


    if args.filename not in ['', "", 'none']:
        print('tensorboard file is saved in', fname)
        writer.close()
    return stopper.best_test_score, stopper.best_epoch, training_time


if __name__ == "__main__":
    from parser import *

    args = get_parser()
    print(args)
    elapsed_times = []
    seed_nums = list(range(10))
    if args.debug:
        seed_nums = [args.runseed]
        if 'inference' in args.tag:
            seed_nums = [42]
    ## 10 random seeds
    results = []
    for seed_num in seed_nums:
        print(f"seed:{seed_num}/{seed_nums}")
        args.runseed = seed_num
        setup_seed(seed_num)
        test_acc, best_epoch, training_time_epoch = main(args)
        results.append(test_acc)
        print(f'avg_test_acc={sum(results) / len(results):.5f}')
        elapsed_times.append(training_time_epoch.sum_elapsed_time())
        print(f'Seed {seed_num}/{max(seed_nums)} Acc array: ', results)
        print(args)

    results = np.array(results)
    elapsed_times = np.array(elapsed_times)
    print(f'avg_test_acc={results.mean():.5f}')
    print(f"avg_test_acc={results.mean() * 100:.3f}$\pm${results.std() * 100:.3f}\n  "
          f"elapsed_times:{elapsed_times.mean():.4f}+-{elapsed_times.std():.4f}s.")
