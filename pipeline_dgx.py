import cupy as cp
import numpy as np

import argparse
import os

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
    argparser.add_argument("--n_samples", type=int, default=100)
    args = argparser.parse_args()

    X, Y = load_data()

    n_samples = int(min(args.n_samples, X.shape[0]))
    X = X[:n_samples]
    Y = Y[:n_samples]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    X_train = cp.asarray(X_train).reshape(-1, 3, 1024)
    X_test = cp.asarray(X_test).reshape(-1, 3, 1024)
    Y_train = encode_labels(Y_train)

    save_file = f"results_{n_samples}.txt"

    for depth in [2, 5, 8, 10, 15, 30, 50]:
        for gap in [True, False]:
            for max_pool in [True, False]:
                for fix in [True, False]:
                    if gap and max_pool:
                        continue
                    print(f"Depth: {depth}, gap: {gap}, max_pool: {max_pool}, fix: {fix}")

                    if gap and max_pool:
                        raise ValueError("Cannot use both gap and max_pool")

                    train_coefs, train_inv_coefs = [], []
                    for i, sample in enumerate(X_train):
                        coefs, inv_coefs = compute_sigma(sample, depth, fix=fix)
                        train_coefs.append(coefs)
                        train_inv_coefs.append(inv_coefs)
                        if i % 10 == 0:
                            print(f"Train coefs: {i/len(X_train)*100:.2f}%", end="\r")


                    test_coefs, test_inv_coefs = [], []
                    for i, sample in enumerate(X_test):
                        coefs, inv_coefs = compute_sigma(sample, depth, fix=fix)
                        test_coefs.append(coefs)
                        test_inv_coefs.append(inv_coefs)
                        if i % 10 == 0:
                            print(f"Test coefs: {i/len(X_test)*100:.2f}%", end="\r")

                    train_kernel = np.zeros((len(X_train), len(X_train)))

                    for i, sample1 in enumerate(X_train):
                        for j, sample2 in enumerate(X_train):
                            train_kernel[i, j] = compute_cross_sigma(
                                sample1, sample2, train_coefs[i], train_coefs[j], train_inv_coefs[i], train_inv_coefs[j], depth, gap=gap, max_pool=max_pool
                            )
                        if i % 10 == 0:
                            print(f"Train kernel: {i/len(X_train)*100:.2f}%", end="\r")

                    test_kernel = np.zeros((len(X_test), len(X_train)))

                    for i, sample1 in enumerate(X_test):
                        for j, sample2 in enumerate(X_train):
                            test_kernel[i, j] = compute_cross_sigma(
                                sample1, sample2, test_coefs[i], train_coefs[j], test_inv_coefs[i], train_inv_coefs[j], depth, gap=gap, max_pool=max_pool
                            )
                        if i % 10 == 0:
                            print(f"Test kernel: {i/len(X_test)*100:.2f}%", end="\r")

                    network = np.linalg.solve(train_kernel, Y_train)

                    predictions = np.matmul(test_kernel, network)
                    predictions = np.argmax(predictions, axis=1)
                    accuracy = np.mean(predictions == Y_test)

                    if not os.path.exists(save_file):
                        with open(save_file, "w") as f:
                            f.write("depth, gap, max_pool, fix, n_samples, accuracy\n")

                    with open(save_file, "a") as f:
                        f.write(f"{depth}, {gap}, {max_pool}, {fix}, {n_samples}, {accuracy}\n")
                    print(f"Accuracy: {accuracy}")
