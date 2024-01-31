import sys

import cupy as cp
import numpy as np

from tqdm import tqdm

from kernels.activation import FastReLU
from kernels.convolution import Convolution4D
from layer import Vanilla, GlobbalAveragePooling

conv = Convolution4D()
relu = FastReLU()


final_layers = {"vanilla": Vanilla(), "gap": GlobbalAveragePooling()}


def compute_sigma(sample, depth):
    coefs, inv_coefs = [1.0], [1.0]

    sigma = cp.einsum("cij,ckl->ijkl", sample, sample)
    print(sigma)
    conv.compiled_kernel(conv.conv_blocks, conv.conv_threads, (sigma,sigma))

    print(sigma)

    theta = sigma.copy()

    for d in range(1, depth):
        K, K_dot, coef = relu(sigma, theta)

        if d != depth - 1:
            sigma = conv(K)
            theta = conv(K_dot)

        coefs.append(coef)
        inv_coefs.append(1.0 / coef)

    return coefs, inv_coefs

def compute_cross_sigma(sample1, sample2, coefs1, coefs2, depth):
    sigma = cp.einsum("cij,ckl->ijkl", sample1, sample2)
    sigma = conv(sigma)

    theta = sigma.copy()

    for d in range(1, depth):
        K, K_dot = relu(sigma, theta, coefs1[d], coefs2[d])

        if d != depth - 1:
            sigma = conv(K)
            theta = conv(K_dot)

    return theta


def encode_labels(labels):
    """Encode the labels.

    Returns:
        numpy.ndarray: The solution of the linear system.
    """
    targets = np.ones((len(labels), 10)) * -1.0
    for i, label in enumerate(labels):
        targets[i, label] = 0.9

    return targets


class ConvNTK:
    """Convolutional Neural Tangent Kernel.

    This class computes the NTK for a convolutional neural network for given depth of convolutional layers.
    The kernel is computed for a given final layer which is specified by the final_layer_name parameter and can be chosen from
        - vanilla: Vanilla layer
        - gap: Global Average Pooling layer
        - max: Max Pooling layer
    """

    def __init__(self, depth=10, final_layer_name="vanilla", verbose=True):
        """Initialize the ConvNTK class.

        The initialisation phase will preprocess the kernel by computing the
        diagonal terms of the covariance matrix for each sample.

        Args:
            depth (int): The depth of the convolutional layers.
        """
        self.depth = depth
        self.final_layer_name = final_layer_name

        self.verbose = verbose


    def train(self, train_samples, train_labels):
        """Train the network by computing the Convolutional Neural Tangent Kernel
        of the training set and solving the kernel regression linear system associated.

        Args:
            train_samples (numpy.ndarray): The training samples.
            train_labels (numpy.ndarray): The training labels.
        """
        self.train_samples = cp.asarray(train_samples)
        self.train_labels = train_labels
        self.train_labels_encoded = encode_labels(self.train_labels)

        if self.verbose: print("Preprocessing the kernel...")
        self.train_coefs, _ = self._preprocess(self.train_samples)
        if self.verbose: print("Computing the kernel...")
        self.train_kernel = self._compute_kernel(self.train_samples, self.train_coefs)
        print('Kernel shape: ', self.train_kernel.shape)

        if self.verbose: print("Solving the linear system...")
        self.network = np.linalg.solve(self.train_kernel.get(), self.train_labels_encoded)
        print('Network shape: ', self.network.shape)

    def evaluate(self, test_samples, test_labels):
        """Evaluate the network by computing the Convolutional Neural Tangent Kernel of
        the test set and computing the accuracy of the network.

        Args:
            test_samples (numpy.ndarray): The test samples.
            test_labels (numpy.ndarray): The test labels.

        Returns:
            float: The accuracy of the network.
        """
        self.test_samples = cp.asarray(test_samples)
        self.test_labels = test_labels
        self.test_labels_encoded = encode_labels(self.test_labels)

        if self.verbose: print("Evaluating the network...")

        if self.verbose: print("Preprocessing the kernel...")
        test_coefs, _ = self._preprocess(test_samples)
        if self.verbose: print("Computing the kernel...")
        self.test_kernel = self._compute_kernel(self.train_samples, self.train_coefs, test_samples, test_coefs)
        print('Kernel shape: ', self.test_kernel.shape)

        if self.verbose: print("Predicting...")
        predictions = np.dot(self.test_kernel.get().T, self.network)
        predictions = np.argmax(predictions, axis=1)

        return np.mean(predictions == self.test_labels_encoded)


    def _compute_kernel(self, samples, coefs, samples2=None, coefs2=None):
        """Compute the kernel for the given final layer.

        For each pair of samples the kernel is computed by doing a "forward" pass in the infinite width limit.

        Args:
            final_layer_name (str): The name of the final layer.
        """
        if samples2 is None: samples2 = samples
        if coefs2 is None: coefs2 = coefs

        kernel = cp.zeros((len(samples), len(samples2)))
        for i, sample1 in tqdm(enumerate(samples), total=len(samples)):
            for j, sample2 in enumerate(samples2):
                features = compute_cross_sigma(
                    sample1, sample2, coefs[i], coefs2[j], self.depth
                )
                kernel[i, j] = final_layers[self.final_layer_name](features)

        return kernel

    def _preprocess(self, samples):
        """Preprocess the kernel.

        Compute the diagonal terms of the covariance matrix for each sample.
        """
        all_coefs, all_inv_coefs = [], []

        for sample in samples:
            sample = cp.asarray(sample)

            coefs, inv_coefs = compute_sigma(sample, self.depth)

            print(coefs)
            sys.exit()

            all_coefs.append(coefs)
            all_inv_coefs.append(inv_coefs)

        return all_coefs, all_inv_coefs
