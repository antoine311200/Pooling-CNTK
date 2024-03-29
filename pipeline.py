import cupy as cp
import numpy as np

from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt

from kernels.utils import (
    Conv,
    conv_blocks,
    conv_threads,
    Relu,
    relu_blocks,
    relu_threads,
)
from data import load_data

from sklearn.model_selection import train_test_split


def compute_sigma(sample, depth, fix=False):
    coefs, inv_coefs = [1.0], [1.0]

    S = cp.matmul(sample.T, sample).reshape(32, 32, 32, 32)
    Conv(conv_blocks, conv_threads, (S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)
    if not fix:
        T += S

    for i in range(1, depth - 1):
        coef = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
        inv_coef = 1.0 / coef
        coefs.append(coef)
        inv_coefs.append(inv_coef)
        Relu(relu_blocks, relu_threads, (S, T, coef, coef, inv_coef, inv_coef))
        Conv(conv_blocks, conv_threads, (S, S))
        Conv(conv_blocks, conv_threads, (T, T))

    coef = cp.sqrt(cp.diag(S.reshape(1024, 1024)).reshape(32, 32))
    inv_coef = 1.0 / coef
    coefs.append(coef)
    inv_coefs.append(inv_coef)
    Relu(relu_blocks, relu_threads, (S, T, coef, coef, inv_coef, inv_coef))

    if fix:
        T -= S
    return coefs, inv_coefs


def compute_cross_sigma(
    sample1,
    sample2,
    coefs1,
    coefs2,
    inv_coefs1,
    inv_coefs2,
    depth,
    fix=False,
    gap=False,
    max_pool=False,
):

    S = cp.matmul(sample1.T, sample2).reshape(32, 32, 32, 32)
    Conv(conv_blocks, conv_threads, (S, S))
    T = cp.zeros((32, 32, 32, 32), dtype=cp.float32)
    if not fix:
        T += S

    for i in range(1, depth - 1):
        Relu(
            relu_blocks,
            relu_threads,
            (S, T, coefs1[i], coefs2[i], inv_coefs1[i], inv_coefs2[i]),
        )
        Conv(conv_blocks, conv_threads, (S, S))
        Conv(conv_blocks, conv_threads, (T, T))

    Relu(
        relu_blocks,
        relu_threads,
        (S, T, coefs1[-1], coefs2[-1], inv_coefs1[-1], inv_coefs2[-1]),
    )

    if fix:
        T -= S
    return cp.mean(T) if gap else cp.max(T) if max_pool else cp.trace(T.reshape(1024, 1024))

def encode_labels(labels):
    targets = np.ones((len(labels), 10)) * -0.1
    for i, label in enumerate(labels):
        targets[i, label] = 0.9

    return targets

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--depth", type=int, default=5)
    argparser.add_argument("--gap", action="store_true")
    argparser.add_argument("--max_pool", action="store_true")
    argparser.add_argument("--fix", action="store_true")
    argparser.add_argument("--n_samples", type=int, default=100)
    args = argparser.parse_args()

    X, Y = load_data()

    X = X[:min(args.n_samples, X.shape[0])]
    Y = Y[:min(args.n_samples, Y.shape[0])]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    X_train = cp.asarray(X_train).reshape(-1, 3, 1024)
    X_test = cp.asarray(X_test).reshape(-1, 3, 1024)
    Y_train = encode_labels(Y_train)

    # depth = args.depth
    gap = args.gap
    max_pool = args.max_pool
    fix = args.fix

    accuracies = []

    for depth in [2, 5, 8, 10, 15]:

        if gap and max_pool:
            raise ValueError("Cannot use both gap and max_pool")

        train_coefs, train_inv_coefs = [], []
        for sample in tqdm(X_train):
            coefs, inv_coefs = compute_sigma(sample, depth, fix=fix)
            train_coefs.append(coefs)
            train_inv_coefs.append(inv_coefs)

        test_coefs, test_inv_coefs = [], []
        for sample in tqdm(X_test):
            coefs, inv_coefs = compute_sigma(sample, depth, fix=fix)
            test_coefs.append(coefs)
            test_inv_coefs.append(inv_coefs)

        train_kernel = np.zeros((len(X_train), len(X_train)))

        for i, sample1 in tqdm(enumerate(X_train), total=len(X_train)):
            for j, sample2 in enumerate(X_train):
                train_kernel[i, j] = compute_cross_sigma(
                    sample1, sample2, train_coefs[i], train_coefs[j], train_inv_coefs[i], train_inv_coefs[j], depth, gap=gap, max_pool=max_pool
                )

        test_kernel = np.zeros((len(X_test), len(X_train)))

        for i, sample1 in tqdm(enumerate(X_test), total=len(X_test)):
            for j, sample2 in enumerate(X_train):
                test_kernel[i, j] = compute_cross_sigma(
                    sample1, sample2, test_coefs[i], train_coefs[j], test_inv_coefs[i], train_inv_coefs[j], depth, gap=gap, max_pool=max_pool
                )


        # print(Y_test)

        network = np.linalg.solve(train_kernel, Y_train)

        print("Network shape: ", network.shape)
        print("Train kernel shape: ", train_kernel.shape)
        print("Test kernel shape: ", test_kernel.shape)

        predictions = np.matmul(test_kernel, network)
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == Y_test)

        print("Accuracy: ", accuracy)
        # print("Predictions: ", predictions)
        # print("Y_test: ", Y_test)

        accuracies.append(accuracy)

    print("Accuracies: ", accuracies)