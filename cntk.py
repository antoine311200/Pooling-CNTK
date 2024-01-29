import cupy as cp
import numpy as np

from kernels.activation import FastReLU
from kernels.convolution import Convolution4D
from layer import Vanilla, GlobbalAveragePooling

conv = Convolution4D()
relu = FastReLU()


final_layers = {"vanilla": Vanilla(), "gap": GlobbalAveragePooling()}


class ConvNTK:
    def __init__(self, samples, labels, depth=10):
        self.samples = samples
        self.labels = labels
        self.depth = depth

        self.kernel = cp.zeros((len(samples), len(samples)))
        self.preprocess()

    def solve(self, final_layer_name):
        self.compute_kernel(final_layer_name)

        self.targets = np.ones((len(self.samples), 10)) * -1.0
        for i, label in enumerate(self.labels):
            self.targets[i, label] = 0.9

        # Need to compute for train and test sets
        # Then compute the linear system solution

    def compute_kernel(self, final_layer_name):
        for i, sample1 in enumerate(self.samples):
            for j, sample2 in enumerate(self.samples):
                features = self.compute_cross_sigma(
                    sample1, sample2, self.all_coefs[i], self.all_coefs[j], self.depth
                )
                self.kernel[i, j] = final_layers[final_layer_name](features)

    def preprocess(self):
        self.all_coefs, self.all_inv_coefs = [], []

        for sample in self.samples:
            sample = cp.asarray(sample)

            coefs, inv_coefs = self.compute_sigma(sample, self.depth)

            self.all_coefs.append(coefs)
            self.all_inv_coefs.append(inv_coefs)

    def compute_sigma(sample, depth):
        coefs, inv_coefs = [1.0], [1.0]

        sigma = cp.einsum("cij,ckl->ijkl", sample, sample)
        sigma = conv(sigma)

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
