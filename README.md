# Pooling Convolutional Neural Tangent Kernel(P-CNTK)

Implementation and extension of the paper "On Exact Computation with an Infinitely Wide Neural Net" by Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang where we derived CNTK formula for other pooling layers.

### Installation

This code is optimised using CuPy which relies on Cuda and GPU computing.

For Cuda 11.x, use the following command to install CuPy:
```sh
$ pip install pip install cupy-cuda11x
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

For Cuda 12.x,
```sh
$ pip install pip install cupy-cuda12x
python -m cupyx.tools.install_library --cuda 12.x --library cutensor
```
