# DRL_Group_Project

Our modification of the paper **"Multi-game Decision Transformers"** by Lee et al. (2022).  
Our code is adapted from the Jupyter notebook provided by the authors at:  
[https://github.com/google-research/google-research/tree/master/multi_game_dt](https://github.com/google-research/google-research/tree/master/multi_game_dt)

Due to outdated dependencies, we ported the notebook into a standalone Python script (`main.py`) which is setup via a shell script (`setup.sh`).

---

## Setup Guide (Or use the Jupyter notebook)

1. **Prerequisites**
   - Ensure that **Anaconda** or **Miniconda** is installed on your system.
   module load anaconda3

2. **Run the Setup Script**
   ```bash
   bash setup.sh
   ```
   This will:
   - Create a conda environment named **`gymenv`** with all necessary dependencies.
3. **Update Import Statements for Stable Baselines 3**

   The original MGDT code used deprecated `baselines.common.atari_wrappers`.  
   In this project, we switch to **Stable Baselines 3** wrappers instead.

   Update any import lines to:

   ```python
   from stable_baselines3.common import atari_wrappers
    
---

## Run Guide (Or use the notebook)

1. **Activate the Environment**
   ```bash
   conda activate gymenv
   ```

2. **Run the Model**
   ```bash
   python main.py
   ```

   By default, the model:
   - Loads cartpole weights (can change)
   - Runs for **200 steps**
   - Uses **10 parallel environments**
   - Runs on all Cartpole control environment (can change)

4. **Modify Parameters**
   To change the training configuration, open `main.py` and edit:
   ```python
   num_steps = 200        # total number of time steps
   num_envs = 10            # number of parallel environments
   ```

## Reference

**Lee, K., Laskin, M., & Srinivas, A. (2022).**  
*Multi-Game Decision Transformers.* Google Research.  
[https://github.com/google-research/google-research/tree/master/multi_game_dt](https://github.com/google-research/google-research/tree/master/multi_game_dt)
