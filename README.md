# DRL_Group_Project

Our modification of the paper **"Multi-game Decision Transformers"** by Lee et al. (2022).  
Our code is adapted from the Jupyter notebook provided by the authors at:  
[https://github.com/google-research/google-research/tree/master/multi_game_dt](https://github.com/google-research/google-research/tree/master/multi_game_dt)

Due to outdated dependencies, we ported the notebook into a standalone Python script (`main.py`) which is setup via a shell script (`setup.sh`).

---

## Setup Guide

1. **Prerequisites**
   - Ensure that **Anaconda** or **Miniconda** is installed on your system.
   module load anaconda3

2. **Run the Setup Script**
   ```bash
   bash setup.sh
   ```
   This will:
   - Create a conda environment named **`gymenv`** with all necessary dependencies.
   - Download the pretrained weights (~1.5 GB) locally. THESE WEIGHTS ARE NOT PRE-TRAINED. The pre-trained weights are available in an uploaded .zip file.

   > *Note:*  
   > The original paper did not provide explicit dependency versions.  
   > All versions were determined through trial and error, so some library compatibility issues may occur.

---

## Run Guide

1. **Activate the Environment**
   ```bash
   conda activate gymenv
   ```

2. **Run the Model**
   ```bash
   python main.py
   ```

   By default, the model:
   - Runs for **200 steps**
   - Uses **10 parallel environments**
   - Runs on all 3 control environments

3. **Modify Parameters**
   To change the training configuration, open `main.py` and edit:
   ```python
   num_steps = 200        # total number of time steps
   num_envs = 10            # number of parallel environments
   ```

4. **Plotting**
   Plotting requires fine-tuning data, which is not available in this repo
   due to dependency conflicts. Please use the list output from the fine-tuning
   repository to print the plots.

---

## File Overview

| File | Description |
|------|--------------|
| `setup.sh` | Creates the conda environment and installs dependencies |
| `main.py` | Runs the MGDT model with customizable parameters |
| `output.txt` | Contains the console log of an example rollout |
| `Breakout_scores.png` | Plot of the model’s performance scores in Breakout |
| `README.md` | Documentation for setup and usage |

---

## Example Rollout and Results

We have included a **rollout log** (5000 steps, 4 envs, Atari Breakout) and its corresponding **performance plot** for reference:

- **[output.txt](./output.txt)** — This file records the console output during a full example rollout of the pretrained model.
- **[Breakout_scores.png](./Breakout_scores.png)** — This image visualizes the final performance metrics from the example rollout.

---

## Reference

**Lee, K., Laskin, M., & Srinivas, A. (2022).**  
*Multi-Game Decision Transformers.* Google Research.  
[https://github.com/google-research/google-research/tree/master/multi_game_dt](https://github.com/google-research/google-research/tree/master/multi_game_dt)
