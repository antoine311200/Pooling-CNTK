#!/bin/bash
python3.10 -m venv .env
source .env/bin/activate
pip install pip install cupy-cuda11x
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
pip install tqdm scikit-learn matplotlib