import numpy as np
import matplotlib.pyplot as plt

arr_cartpole_pre = np.array([40., 25., 10.,  9., 36., 29.,  9.,  9., 11.,  9.])
arr_mountaincar_pre = np.array([-200. for i in range(10)])
arr_pendulum_pre = np.array([ -824.,  -904.,  -983.,  -963., -1463., -1077., -1447., -1354.,  -914., -1422.])

arr_cartpole_post = np.array([])
arr_mountaincar_post = np.array([])
arr_pendulum_post = np.array([])

plt.plot(arr_pendulum_pre, 'o')
plt.title(f'Scores for Pendulum pre-FT')
plt.xlabel('trial index')
plt.ylabel('score')
plt.savefig(f"Pendulum_pre_scores.png", dpi=300, bbox_inches='tight')