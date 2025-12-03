import numpy as np
import matplotlib.pyplot as plt

arr_cartpole_pre = np.array([8.0, 10.0, 19.0, 16.0, 32.0, 10.0, 8.0, 16.0, 10.0, 11.0])
arr_mountaincar_pre = np.array([-200. for i in range(10)])
arr_pendulum_pre = np.array([-1180.0, -1232.0, -1213.0, -1182.0, -1209.0, -1263.0, -1549.0, -1027.0, -1547.0, -1059.0])

arr_cartpole_post = np.array([])
arr_mountaincar_post = np.array([])
arr_pendulum_post = np.array([])

# --------------- BEFORE FINE-TUNING -----------------

plt.plot(arr_cartpole_pre, 'o')
plt.title(f'Scores for CartPole pre-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"CartPole_pre_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_mountaincar_pre, 'o')
plt.title(f'Scores for MountainCar pre-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"MountainCar_pre_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_pendulum_pre, 'o')
plt.title(f'Scores for Pendulum pre-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"Pendulum_pre_scores.png", dpi=300, bbox_inches='tight')

plt.clf()
print("Pre-plots done")

# --------------- AFTER FINE-TUNING -----------------

plt.plot(arr_pendulum_pre, 'o')
plt.title(f'Scores for CartPole post-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"CartPole_post_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_mountaincar_pre, 'o')
plt.title(f'Scores for MountainCar post-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"MountainCar_post_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_pendulum_pre, 'o')
plt.title(f'Scores for Pendulum post-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"Pendulum_post_scores.png", dpi=300, bbox_inches='tight')

plt.clf()
print("Post-plots done")


# --------------- COMBINED -----------------

plt.plot(arr_cartpole_pre, 'o', color='red')
plt.plot(arr_cartpole_post, 'o', color='blue')
plt.title(f'Combined Scores for CartPole')
plt.xlabel('trial index')
plt.ylabel('score')
plt.legend()
plt.savefig(f"CartPole_comb_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_mountaincar_pre, 'o', color='red')
plt.plot(arr_mountaincar_post, 'o', color='blue')
plt.title(f'Combined Scores')
plt.xlabel('trial index')
plt.ylabel('score')
plt.legend()
plt.savefig(f"MountainCar_comb_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_pendulum_pre, 'o', color='red')
plt.plot(arr_pendulum_post, 'o', color='blue')
plt.title(f'Combined Score')
plt.xlabel('trial index')
plt.ylabel('score')
plt.legend()
plt.savefig(f"Pendulum_comb_scores.png", dpi=300, bbox_inches='tight')

plt.clf()
print("Combined plots done")
