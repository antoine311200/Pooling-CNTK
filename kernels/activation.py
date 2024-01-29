import cupy as cp

class FastReLU:
    """FastReLU kernel for 4D tensors.

    The operations performed is to compute the two following quantities:

    (1)     K^(h)(x_a, x_b)_ijkl = coef * E[ReLU(u) * ReLU(v)] where u, v are drawn from N(0, Lambda^(h)_ijkl(x_a, x_b))

    (2)     K_dot^(h)(x_a, x_b)_ijkl = coef * E[ReLU_dot(u) * ReLU_dot(v)] where u, v are drawn from N(0, Lambda^(h)_ijkl(x_a, x_b))
    """

    def __init__(self):
        kernel_code = '''
        extern "C" __global__ void activation(
            float s[32][32][32][32],
            float t[32][32][32][32],
            const float l[32][32],
            const float r[32][32],
            const float il[32][32],
            const float ir[32][32]
        ) {
            int x1 = blockIdx.x;
            int y1 = blockIdx.y;
            int x2 = threadIdx.x + ((blockIdx.z >> 2) << 3);
            int y2 = threadIdx.y + ((blockIdx.z & 3) << 3);

            float S = s[x1][y1][x2][y2];
            float T = t[x1][y1][x2][y2];
            float L = l[x1][y1];
            float R = r[x2][y2];
            float iL = il[x1][y1];
            float iR = ir[x2][y2];

            S = S * iL * iR;

            float BS = (S * (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) + sqrtf(1.0f - min(S * S, 1.0f))) * L * R / 28.274333882308138f;

            S = (3.141592654f - acosf(max(min(S, 1.0f), -1.0f))) / 28.274333882308138;

            t[x1][y1][x2][y2] = T * S + BS;
            s[x1][y1][x2][y2] = BS;

        }'''
        self.compiled_kernel = cp.RawKernel(kernel_code, "activation")

        self.trans_blocks = (32, 32, 16)
        self.trans_threads = (8, 8)

    def __call__(self, sigma, H, coef1=None, coef2=None):
        """Apply the activation function to the input

        Args:
            sigma (cupy.ndarray): Input to the activation function
            H (cupy.ndarray): Input to the activation function

        Returns:
            sigma (cupy.ndarray): Input to the activation function
            H (cupy.ndarray): Input to the activation function
            coef (cupy.ndarray): Input to the activation function
        """
        # Convolution blocks and threads

        sigma = cp.asarray(sigma).copy()
        H = cp.asarray(H).copy()

        if coef1 is None or coef2 is None:
            coef = cp.sqrt(cp.diag(sigma.reshape(1024, 1024)).reshape(32, 32))
            self.compiled_kernel(self.trans_blocks, self.trans_threads, (sigma, H, coef, coef, 1. / coef, 1. / coef))

            return sigma, H, coef
        else:
            self.compiled_kernel(self.trans_blocks, self.trans_threads, (sigma, H, coef1, coef2, 1. / coef1, 1. / coef2))

            return sigma, H
