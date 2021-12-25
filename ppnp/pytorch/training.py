from typing import Type
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ..data.sparsegraph import SparseGraph
from ..preprocessing import gen_seeds, gen_splits, normalize_attributes
from .earlystopping import EarlyStopping, stopping_args
from .utils import matrix_to_torch, sparse_matrix_to_torch
from sklearn.metrics import f1_score
from .propagation import calc_A_hat
from layers import SparseMM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np.astype(np.int32))
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
                   for phase, dataset in datasets.items()}
    return dataloaders

def train_model(
        name: str, model_name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        learning_rate: float, reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10, early: bool = True) -> nn.Module:
    labels_all = graph.labels
    idx_np = {}
    # (1) split labels to train, stopping, val/test
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(labels_all, idx_split_args, test=test) #
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed) # random
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1] # cora: 2879
    nclasses = max(labels_all) + 1 # cora: 7
    if model_name == 'GCN':
        model = model_class(nfeat=nfeatures, nhid=16, nclass=nclasses, dropout=model_args['drop_prob'], dropout_adj=model_args['dropoutadj_GCN'], layer=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'], weight_decay=model_args['weight_decay'])
    else: # APPNP
        model = model_class(nfeatures, nclasses, hiddenunits=model_args['hiddenunits'], drop_prob=model_args['drop_prob'], propagation=model_args['propagation']).to(device) # MLP network f: X -> H
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize all network

    reg_lambda = torch.tensor(reg_lambda, device=device)
    # (2) optimizer, dataloader, early_stopping
    dataloaders = get_dataloaders(idx_all, labels_all) # random index to dataloaders
    early_stopping = EarlyStopping(model, **stopping_args)
    # nomalize features and then to tensor
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    attr_mat_norm = matrix_to_torch(attr_mat_norm_np).to(device)

    # (3) training:
    if early:
        epoch_stats = {'train': {}, 'stopping': {}}
    else:
        epoch_stats = {'train': {}}
    start_time = time.time()
    last_time = start_time
    for epoch in range(early_stopping.max_epochs): # 10000
        for phase in epoch_stats.keys(): # 2 phases: train, stopping, train 1 epoch, evaluate on stopping dataset
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            for idx, labels in dataloaders[phase]: # training set / early stopping set
                idx = idx.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'): # train: True
                    if model_name == 'GCN':
                        adj = sparse_matrix_to_torch(calc_A_hat(graph.adj_matrix)).to(device)
                        log_preds = model(attr_mat_norm, adj, idx) # A is in model's buffer
                    else:
                        log_preds = model(attr_mat_norm, idx)  # A is in model's buffer

                    preds = torch.argmax(log_preds, dim=1)

                    # Calculate loss
                    cross_entropy_mean = F.nll_loss(log_preds, labels)
                    if model_name == 'GCN':
                        loss = cross_entropy_mean
                    else:
                        l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                        loss = cross_entropy_mean + reg_lambda / 2 * l2_reg # cross loss + L2 regularization

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Collect statistics
                    running_loss += loss.item() * idx.size(0)
                    running_corrects += torch.sum(preds == labels)

            # Collect statistics (current epoch)
            epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
            epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

        # print logging each interval
        if epoch % print_interval == 0:
            duration = time.time() - last_time  # each interval including training and early-stopping
            last_time = time.time()
            if early:
                logging.info(f"Epoch {epoch}: "
                             f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                             f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                             f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                             f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                             f"({duration:.3f} sec)")
            else:
                logging.info(f"Epoch {epoch}: "
                             f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                             f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                             f"({duration:.3f} sec)")
        # (4) check whether it stops on some epoch
        if early:
            if len(early_stopping.stop_vars) > 0:
                stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars] # 'acc', 'loss'
                if early_stopping.check(stop_vars, epoch): # whether exist improvement for patience times
                    break
    runtime = time.time() - start_time
    logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")

    # (5) evaluate the best model from early stopping on test set
    # Load best model weights
    if early:
        model.load_state_dict(early_stopping.best_state)

    stopping_preds = get_predictions(model, model_name, attr_mat_norm, graph.adj_matrix, idx_all['stopping'])
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    stopping_f1 = f1_score(stopping_preds, labels_all[idx_all['stopping']], average='micro')
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    valtest_preds = get_predictions(model, model_name, attr_mat_norm, graph.adj_matrix, idx_all['valtest'])
    valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_f1 = f1_score(valtest_preds, labels_all[idx_all['valtest']], average='micro')
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")

    # (6) return result
    result = {}
    result['early_stopping'] = {'accuracy': stopping_acc, 'f1_score': stopping_f1}
    result['valtest'] = {'accuracy': valtest_acc, 'f1_score': valtest_f1}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime / (epoch + 1)
    return result

def get_predictions(model, model_name, attr_matrix, adj_matrix, idx, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    dataset = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    preds = []
    for idx, in dataloader:
        idx = idx.to(attr_matrix.device)
        with torch.set_grad_enabled(False):
            if model_name == 'GCN':
                adj = sparse_matrix_to_torch(calc_A_hat(adj_matrix)).to(device)
                log_preds = model(attr_matrix, adj, idx)  # A is in model's buffer
            else:
                log_preds = model(attr_matrix, idx)  # A is in model's buffer

            preds.append(torch.argmax(log_preds, dim=1))
    return torch.cat(preds, dim=0).cpu().numpy()


def train_model_AdaGCN(
        name: str, model_name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        learning_rate: float, reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10, early: bool = True) -> nn.Module:
    labels_all = graph.labels.astype(np.int)
    idx_np = {}
    # (1) split labels to train, stopping, val/test
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(labels_all, idx_split_args, test=test)
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed) # random
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1] # cora: 2879
    nclasses = max(labels_all) + 1 # cora: 7
    # define model
    print('device:', device)
    model = model_class(nfeat=nfeatures, nhid=model_args['hid_AdaGCN'], nclass=nclasses, dropout=model_args['drop_prob'], dropout_adj=model_args['dropoutadj_AdaGCN']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'], weight_decay=model_args['weight_decay'])

    # (2) optimizer, dataloader, early_stopping
    dataloaders = get_dataloaders(idx_all, labels_all) # random index to dataloaders
    early_stopping = EarlyStopping(model, **stopping_args)
    # nomalize features and then to tensor
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    # attr_mat_norm = matrix_to_torch(attr_mat_norm_np).to(device) # SparseTensor
    attr_mat_norm = torch.FloatTensor(np.array(attr_mat_norm_np.todense())).to(device) # DenseTensor

    # (3) define variables:
    if early:
        epoch_stats = {'train': {}, 'stopping': {}}
    else:
        epoch_stats = {'train': {}}
    start_time = time.time()
    last_time = start_time

    adj = sparse_matrix_to_torch(calc_A_hat(graph.adj_matrix)).to(device)
    sample_weights = torch.ones(graph.adj_matrix.shape[0])  # 2708
    sample_weights = sample_weights[idx_all['train']]  # 140*1
    sample_weights = sample_weights / sample_weights.sum()  # 1/140
    sample_weights = sample_weights.to(device)
    # global sample_weights
    results = torch.zeros(graph.adj_matrix.shape[0], nclasses).to(device)
    ALL_epochs = 0

    # (4) train AdaGCN
    features = attr_mat_norm
    for layer in range(model_args['layers']):
        logging.info(f"|This is the {layer+1}th layer!")
        # (5) train each classifier:  each epoch: training + early stopping
        for epoch in range(early_stopping.max_epochs): # 10000
            for phase in epoch_stats.keys(): # 2 phases: train, stopping, train 1 epoch, evaluate on stopping dataset
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0
                running_corrects = 0

                for idx, labels in dataloaders[phase]: # training set / early stopping set
                    idx = idx.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'): # train: True
                        log_preds = model(features, adj, idx)  # A is in model's buffer
                        loss = F.nll_loss(log_preds, labels, reduction='none')  # each loss
                        # core 1: weighted loss
                        if phase == 'train':
                            loss = loss * sample_weights
                        preds = torch.argmax(log_preds, dim=1)
                        loss = loss.sum()
                        l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                        loss = loss + reg_lambda / 2 * l2_reg  # cross loss + L2 regularization
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # Collect statistics
                        running_loss += loss.item()
                        # running_loss += loss.item() * idx.size(0)
                        running_corrects += torch.sum(preds == labels)

                # Collect statistics (current epoch)
                epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
                epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

            # print logging each interval
            if epoch % print_interval == 0:
                duration = time.time() - last_time # each interval including training and early-stopping
                last_time = time.time()
                if early:
                    logging.info(f"Epoch {epoch}: "
                                 f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                                 f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                                 f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                                 f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                                 f"({duration:.3f} sec)")
                else:
                    logging.info(f"Epoch {epoch}: "
                                 f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                                 f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                                 f"({duration:.3f} sec)")
            # (4) check whether it stops on some epoch
            if early:
                if len(early_stopping.stop_vars) > 0:
                    stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars] # 'acc', 'loss'
                    if early_stopping.check(stop_vars, epoch): # whether exist improvement for patience times
                        break

        # (6) SAMME.R
        ALL_epochs += epoch
        runtime = time.time() - start_time
        logging.log(22, f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch} ({runtime:.3f} sec)")
        # Load best model weights
        if early:
            model.load_state_dict(early_stopping.best_state)
        model.eval()
        output = model(features, adj, torch.arange(graph.adj_matrix.shape[0])).detach()
        output_logp = torch.log(F.softmax(output, dim=1))
        h = (nclasses - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))
        results += h
        # adjust weights
        temp = F.nll_loss(output_logp[idx_all['train']], torch.LongTensor(labels_all[idx_all['train']].astype(np.int32)).to(device), reduction='none')  # 140*1
        weight = sample_weights * torch.exp((1 - (nclasses - 1)) / (nclasses - 1) * temp)  # update weights
        weight = weight / weight.sum()
        sample_weights = weight.detach()

        # update features
        features = SparseMM.apply(adj, features).detach() # adj: tensor[2810, 2810],  features: tensor[2810,2879]

    # (5) evaluate the best model from early stopping on test set
    runtime = time.time() - start_time
    stopping_preds = torch.argmax(results[idx_all['stopping']], dim=1).cpu().numpy()
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    stopping_f1 = f1_score(stopping_preds, labels_all[idx_all['stopping']], average='micro')
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    valtest_preds = torch.argmax(results[idx_all['valtest']], dim=1).cpu().numpy()
    valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_f1 = f1_score(valtest_preds, labels_all[idx_all['valtest']], average='micro')
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")

    # (6) return result
    result = {}
    result['early_stopping'] = {'accuracy': stopping_acc, 'f1_score': stopping_f1}
    result['valtest'] = {'accuracy': valtest_acc, 'f1_score': valtest_f1}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime / (ALL_epochs + 1)
    return result
