#!/usr/bin/env sh

module load cuda/9.0
module load python3/3.6.2
module load tensorflow/1.10-gpu-python-3.6.2

pip3 install --user keras
pip3 install --user pillow

pip3 install --user matplotlib
