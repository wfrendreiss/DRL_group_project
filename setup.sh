#!/bin/bash
conda create -y -n gymenv python=3.7
conda run -n gymenv pip install "pip==24.0.0"
conda run -n gymenv pip install "setuptools==65.5.0"
conda run -n gymenv pip install "wheel==0.38.0"
conda run -n gymenv pip install "pygame==2.5.2"
conda run -n gymenv pip install "jax==0.3.15"
conda run -n gymenv pip install jaxlib==0.3.22 -f https://storage.googleapis.com/jax-releases/jax_releases.html
conda run -n gymenv pip install "dopamine-rl==2.0.3"
conda run -n gymenv pip install "gym[atari,accept-rom-license]==0.21.0"
conda run -n gymenv pip install --upgrade "jax" "jaxlib"
conda run -n gymenv pip install "dm-haiku==0.0.9"
conda run -n gymenv pip install "tensorflow_datasets==3.2.1"
conda run -n gymenv pip install "tensorflow==1.15.5"
conda run -n gymenv pip install --upgrade "tensorflow==1.15.5" "protobuf<4"
conda run -n gymenv pip install "matplotlib"
conda run -n gymenv pip install atari_py
# conda run -n gymenv pip install gymnasium[classic-control]
conda run -n gymenv pip install "tf-slim==1.1.0"
conda run -n gymenv pip install "importlib-metadata==4.13.0"
conda run -n gymenv pip install "optax==0.1.2"
conda run -n gymenv pip install scipy
conda run -n gymenv pip install "numpy==1.18.5"
conda run -n gymenv pip install "chex==0.1.5"
conda run -n gymenv pip install ipykernel

wget https://storage.googleapis.com/rl-infra-public/multi_game_dt/checkpoint_38274228.pkl -O checkpoint_38274228.pkl

