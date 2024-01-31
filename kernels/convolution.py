import cupy as cp


class Convolution4D:
    """Convolution4D kernel for 4D tensors.

    The convolution defined is this class perform a special action, that is for each element of the output tensor
    at indices (i, j, k, l) the value is the trace of the subtensor for indices in
    D_{i, j, k, l} = {(i+a, j+b, k+c, l+d) | a, b, c, d in {-(q-1)/2, ..., (q-1)/2}},
    that is tr(K_{D_{i, j, k, l}}) = sum_ij K_{D_{i, j, i, j}}.
    with filter size q = 3 and stride 1.

    """

    def __init__(self):
        kernel_code = """
        extern "C" __global__ void convolution4d(const float input[32][32][32][32], float output[32][32][32][32])
        {
            int x1 = threadIdx.x + blockIdx.x - 31;
            int y1 = threadIdx.y + blockIdx.y - 31;
            int x2 = threadIdx.x;
            int y2 = threadIdx.y;

            __shared__ float d[32 + 2][32 + 2];
            if (x2 == 0) {
                d[0][y2 + 1] = d[33][y2 + 1] = 0;
                if (x2 == 0 && y2 == 0) d[0][0] = d[0][33] = d[33][0] = d[33][33] = 0;
            }
            if (y2 == 0) {
                d[x2 + 1][0] = d[x2 + 1][33] = 0;
            }

            if (x1 < 0 || x1 > 31 || y1 < 0 || y1 > 31) {
                d[x2 + 1][y2 + 1] = 0;
                return;
            }
            else d[x2 + 1][y2 + 1] = input[x1][y1][x2][y2];

            __syncthreads();

            output[x1][y1][x2][y2] = d[x2][y2] + d[x2][y2 + 1] + d[x2][y2 + 2]
                            + d[x2 + 1][y2] + d[x2 + 1][y2 + 1] + d[x2 + 1][y2 + 2]
                            + d[x2 + 2][y2] + d[x2 + 2][y2 + 1] + d[x2 + 2][y2 + 2];

        }"""
        self.compiled_kernel = cp.RawKernel(kernel_code, "convolution4d")

        self.conv_blocks = (63, 63)
        self.conv_threads = (32, 32)

    def __call__(self, input_):
        # Convolution blocks and threads

        input = input_.copy()
        output = cp.zeros_like(input)

        self.compiled_kernel(self.conv_blocks, self.conv_threads, (input, output))

        return output