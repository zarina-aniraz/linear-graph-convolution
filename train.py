import torch
from torch import optim
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from ABIDEParser import load_ABIDE, get_labels

from time import perf_counter
import numpy as np
import random

from SGC.utils import sgc_precompute, set_seed
from SGC.models import get_model
from utils import accuracy
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, help="Disables CUDA training.")
parser.add_argument("--seed", type=int, default=123, help="Random seed.")
parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=0,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument("--hidden", type=int, default=0, help="Number of hidden units.")
parser.add_argument("--dropout", type=float, default=0, help="Dropout rate (1 - keep probability).")
parser.add_argument("--degree", type=int, default=2, help="degree of the approximation.")
parser.add_argument("--folds", type=int, default=10, help="k-folds for cross-validation.")
parser.add_argument("--graph_type", type=str, default="original",
                    choices=["original", "graph_no_features", "graph_random", "graph_identity"])
parser.add_argument("--use_relu", action="store_true", default=False, help="Use relu.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def SGC_main(adj, features, labels, epochs, lr, dropout, 
            weight_decay, use_relu, idx_train, idx_val, idx_test):

    model = get_model(
        "SGC",
        features.size(1),
        labels.max().item() + 1,
        args.hidden,
        dropout,
        args.cuda,
        use_relu,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    train_acc = []
    train_loss = []

    def train_regression(model, train_features, train_labels, val_features, val_labels,
                        epochs=args.epochs, weight_decay=args.weight_decay, 
                        lr=args.lr, dropout=args.dropout):

        t = perf_counter()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(train_features)
            loss_train = F.cross_entropy(output, train_labels)
            train_loss.append(loss_train.item())
            loss_train.backward()
            acc_train = accuracy(output, train_labels)
            train_acc.append(acc_train.item())
            optimizer.step()
        train_time = perf_counter() - t

        with torch.no_grad():
            model.eval()
            output = model(val_features)
            acc_val = accuracy(output, val_labels)

        return model, acc_val, train_time, train_acc, train_loss

    def test_regression(model, test_features, test_labels):
        model.eval()
        return accuracy(model(test_features), test_labels)

    model, acc_val, train_time, train_acc, train_loss = train_regression(
        model,
        features[idx_train],
        labels[idx_train],
        features[idx_val],
        labels[idx_val],
    )

    acc_test = test_regression(model, features[idx_test], labels[idx_test])

    return acc_val, acc_test, train_time


def k_fold_run():

    y = get_labels()
    X = np.array([i for i in range(0, len(y))])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
    skf.get_n_splits(X, y)

    adj, features, labels = load_ABIDE(args.graph_type)
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("Pre-compute time: {:.4f}s".format(precompute_time))

    fold_acc_list = []
    fold_time_list = []

    cur_fold = 1

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        idx_train = torch.LongTensor(X_train)
        idx_test = torch.LongTensor(X_test)

        acc_val, acc_test, train_time = SGC_main(adj, features, labels, args.epochs,
                                                args.lr, args.dropout, args.weight_decay,
                                                args.use_relu, idx_train, idx_test, idx_test)

        print("Fold #{:2} - Test Accuracy: {:.2f}%".format(cur_fold, acc_test * 100))
        cur_fold += 1

        fold_acc_list.append(acc_test.item())
        fold_time_list.append(train_time)

    return fold_acc_list, fold_time_list


fold_acc_list, fold_time_list = k_fold_run()
print("Average accuracy over {} folds: {:.2f}%".format(args.folds, np.average(fold_acc_list) * 100))
print("Average time over {} folds: {:.2f}s".format(args.folds, np.average(fold_time_list)))