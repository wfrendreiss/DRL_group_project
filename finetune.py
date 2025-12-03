
# =========================================================
# FULL MGDT FINE-TUNING SCRIPT (OFFLINE, 100k steps)
# =========================================================

import pickle
import optax
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import os
import glob

# =========================================================
# 0. IMPORT YOUR MGDT MODEL
# =========================================================

from main import DecisionTransformer   # modify if path differs
# ^ YOU MUST HAVE THIS FILE — the MGDT model you pasted earlier


# ------------------------------
# 1. Load offline trajectories
# ------------------------------
def load_offline_trajectories(dataset_dir="ppo_datasets"):
    """Load all trajectory datasets from the specified directory."""
    trajectories = []
    dataset_files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
    
    if not dataset_files:
        raise ValueError(f"No .pkl files found in {dataset_dir}/")
    
    print(f"Loading trajectories from {len(dataset_files)} dataset files...")
    
    for dataset_file in sorted(dataset_files):
        print(f"  Loading {dataset_file}...")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
            # If dataset is a list, extend; if it's a single trajectory dict, append
            if isinstance(dataset, list):
                trajectories.extend(dataset)
                print(f"    Added {len(dataset)} trajectories")
            else:
                trajectories.append(dataset)
                print(f"    Added 1 trajectory")
    
    print(f"\nTotal trajectories loaded: {len(trajectories)}")
    if len(trajectories) > 0:
        sample_traj = trajectories[0]
        print(f"Sample trajectory keys: {sample_traj.keys()}")
        if "actions" in sample_traj:
            print(f"Sample trajectory length: {len(sample_traj['actions'])}")
    
    return trajectories

# Load all offline trajectories
trajectories = load_offline_trajectories()


# =========================================================
# 2. DEFINE THE MGDT MODEL + HAIKU TRANSFORM
# =========================================================

NUM_ACTIONS = 18         # CartPole
NUM_REWARDS = 4
RETURN_RANGE = (-20, 100)
D_MODEL = 1280
NUM_LAYERS = 10
CONV_DIM = 256

def forward_fn(inputs, is_training):
    model = DecisionTransformer(
        num_actions=NUM_ACTIONS,
        num_rewards=NUM_REWARDS,
        return_range=RETURN_RANGE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        dropout_rate=0.1,
        predict_reward=True,
        single_return_token=True,
        conv_dim=CONV_DIM,
    )
    return model(inputs, is_training=is_training)


model_fn = hk.transform_with_state(forward_fn)


# =========================================================
# 3. INITIALIZE MODEL (using dummy input)
# =========================================================

dummy_batch = {
    "observations": jnp.zeros([1, 4, 84, 84, 1], dtype=jnp.float32),
    "actions": jnp.zeros([1, 4], dtype=jnp.int32),
    "rewards": jnp.zeros([1, 4], dtype=jnp.int32),
    "returns-to-go": jnp.zeros([1, 4], dtype=jnp.int32),
}

rng = jax.random.PRNGKey(42)
model_params, model_state = model_fn.init(rng, dummy_batch, is_training=True)


# =========================================================
# 4. LOAD YOUR PRETRAINED MGDT CHECKPOINT
# =========================================================

print("Loading pretrained MGDT checkpoint...")
ckpt_params, ckpt_state = pickle.load(open("checkpoint_38274228.pkl", "rb"))
model_params = ckpt_params
model_state = ckpt_state


# =========================================================
# 5. BATCH SAMPLER FROM TRAJECTORIES
# =========================================================

SEQ_LEN = 4
BATCH_SIZE = 256

def sample_batch(trajectories, batch_size=BATCH_SIZE, T=SEQ_LEN):
    ob, ac, rw, rtg = [], [], [], []

    while len(ob) < batch_size:
        traj = np.random.choice(trajectories)
        L = len(traj["actions"])
        if L < T:
            continue
        start = np.random.randint(0, L - T)

        ob.append(traj["observations"][start:start+T])
        ac.append(traj["actions"][start:start+T])
        rw.append(traj["rewards"][start:start+T])
        rtg.append(traj["returns_to_go"][start:start+T])

    batch = {
        "observations": jnp.array(ob),
        "actions": jnp.array(ac, dtype=jnp.int32),
        "rewards": jnp.array(rw, dtype=jnp.int32),
        "returns-to-go": jnp.array(rtg, dtype=jnp.int32),
    }
    return batch


# =========================================================
# 6. LOSS FUNCTION
# =========================================================

def loss_fn(params, state, rng, batch):
    outputs, new_state = model_fn.apply(params, state, rng, batch, is_training=True)
    return outputs["loss"], (outputs, new_state)


# =========================================================
# 7. OPTIMIZER (MGDT FINE-TUNE HYPERPARAMS)
# =========================================================

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.add_decayed_weights(1e-2),
    optax.lamb(learning_rate=1e-4),
)

opt_state = optimizer.init(model_params)


# =========================================================
# 8. TRAIN-STEP (JIT)
# =========================================================

@jax.jit
def train_step(params, state, opt_state, rng, batch):
    print("#1")
    (loss, (outputs, new_state)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params, state, rng, batch)
    print("#2")
    updates, opt_state = optimizer.update(grads, opt_state, params)
    print("#3")
    params = optax.apply_updates(params, updates)
    print("#4")
    return params, new_state, opt_state, loss

def mgdt_action(rng, obs_batch):
    outputs = model.apply(model_params, model_state, rng, obs_batch)
    logits = outputs["action_logits"]
    actions = jnp.argmax(logits, axis=-1)
    return np.array(actions), rng



# =========================================================
# 9. FINE-TUNE LOOP (100k STEPS)
# =========================================================

import gym
from main import _batch_rollout, build_control_env_fn


def fine_tune(trajectories, steps=100_000):
    print("Starting finetuning")
    global model_params, model_state, opt_state
    loss_history = []
    eval_history = []

    env = gym.make("CartPole-v1")
    rng = jax.random.PRNGKey(0)

    for step in range(steps):
        print("step: ", step)
        batch = sample_batch(trajectories)
        rng, key = jax.random.split(rng)
        print("calling train_step")
        model_params, model_state, opt_state, loss = train_step(
            model_params, model_state, opt_state, key, batch
        )
        loss_history.append(float(loss))   # convert JAX scalar to Python float

        if step % 100 == 0:
            print(f"[FT] step={step} loss={float(loss):.4f}")
            # Run evaluation rollout
            print("EVALUATING ROLLOUT")
            env_fn = build_control_env_fn("CartPole")
            env_batch = [env_fn() for _ in range(10)]

            rew_sum, frames, rng = _batch_rollout(
                rng, env_batch, mgdt_action, num_steps=200
            )
            eval_reward = float(np.mean(rew_sum))
            eval_history.append(eval_reward)
    
            print(f"-> Eval return: {eval_reward:.2f}")
        if step % 10_000 == 0 and step > 0:
            with open(f"checkpoint_ft_step{step}.pkl", "wb") as f:
                pickle.dump((model_params, model_state), f)

    with open("checkpoint_ft_final.pkl", "wb") as f:
        pickle.dump((model_params, model_state), f)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MGDT Fine-Tuning Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("finetune_loss_curve.png", dpi=200)
    plt.close()

    print("Saved: finetune_loss_curve.png")

    plt.figure(figsize=(10, 4))
    plt.plot(eval_history, label="Eval Return", color="orange")
    plt.xlabel("Evaluation Step (every 100 iters)")
    plt.ylabel("Return")
    plt.title("MGDT Evaluation Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("finetune_eval_curve.png", dpi=200)
    plt.close()

    print(" Saved: finetune_eval_curve.png")


    print("DONE — fine-tuned MGDT saved!")


if __name__ == "__main__":
    fine_tune(trajectories)
