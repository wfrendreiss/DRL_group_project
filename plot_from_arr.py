import numpy as np
import matplotlib.pyplot as plt

arr_cartpole_pre = np.array([8.0, 10.0, 19.0, 16.0, 32.0, 10.0, 8.0, 16.0, 10.0, 11.0])
arr_mountaincar_pre = np.array([-200. for i in range(10)])
arr_pendulum_pre = np.array([-1180.0, -1232.0, -1213.0, -1182.0, -1209.0, -1263.0, -1549.0, -1027.0, -1547.0, -1059.0])

arr_cartpole_post = np.array([10, 11, 14, 8, 8, 11, 13, 8, 29, 16])
arr_mountaincar_post = np.array([])
arr_pendulum_post = np.array([-1130.0, -1141.0, -1151.0, -1508.0, -1133.0, -1189.0, -871.0, -1014.0, -1569.0, -1280.0])

print(np.mean(arr_cartpole_post))
print(np.std(arr_cartpole_post))
exit()

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

plt.plot(arr_cartpole_post, 'o')
plt.title(f'Scores for CartPole post-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"CartPole_post_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_pendulum_post, 'o')
plt.title(f'Scores for Pendulum post-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"Pendulum_post_scores.png", dpi=300, bbox_inches='tight')

plt.clf()
print("Post-plots done")


# --------------- COMBINED -----------------

plt.plot(arr_cartpole_pre, 'o', color='red', label="Before FT")
plt.plot(arr_cartpole_post, 'o', color='blue', label="After FT")
plt.title(f'Combined Scores for CartPole')
plt.xlabel('trial index')
plt.ylabel('score')
plt.legend()
plt.savefig(f"CartPole_comb_scores.png", dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(arr_pendulum_pre, 'o', color='red', label="Before FT")
plt.plot(arr_pendulum_post, 'o', color='blue', label="After FT")
plt.title(f'Combined Scores for Pendulum')
plt.xlabel('trial index')
plt.ylabel('score')
plt.legend()
plt.savefig(f"Pendulum_comb_scores.png", dpi=300, bbox_inches='tight')

plt.clf()
print("Combined plots done")
