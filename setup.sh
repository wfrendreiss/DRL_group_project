#!/bin/bash
conda create -y -n gymenv python=3.12
conda run -n gymenv pip install jax
conda run -n gymenv pip install tensorflow
conda run -n gymenv pip install tensorflow_datasets
conda run -n gymenv pip install gym
conda run -n gymenv pip install gymnasium
conda run -n gymenv pip install matplotlib
conda run -n gymenv pip install opencv-python
conda run -n gymenv pip install dopamine-rl
conda run -n gymenv pip install stable-baselines3
conda run -n gymenv pip install dm-haiku

