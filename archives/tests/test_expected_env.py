#%%
from RLE.env.RLEEnv_expectation import RLEEnv_expectation
import gymnasium as gym
import numpy as np

env_config = {"D":6, "N_cohort":3, "N_total":36, "phi":0.25, "early_stop":True}
env = gym.make("RLEEnv_expectation", config=env_config)
env.reset()
for i in range(11):
    if i<10:
        action = np.random.randint(0, 3)
    else:
        action = np.random.randint(3, env.D+2)
    observation, reward, terminated, truncated, info = env.step(action)
    # print(action)
    # print(f"{observation=}, {reward=}, {terminated=}, {truncated=}, {info=}")
    if i ==9:
        state = env.get_state()
        obs = env.last_obs
        true_MTD = env.MTD_true
        true_index = [index for index, mtd in enumerate(env.MTD_trues) if mtd==true_MTD]
# %%
env.reset()
env.set_state(state)

print(obs)
print(true_index)
# %%
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.plot(np.arange(20), env.posteriors)
# plt.show()
# %%
# rand_index = np.random.randint(20)
# scenario = env.scenarios[rand_index]
# MTD_true = env.MTD_trues[rand_index]
# Ns = [10]*env.D
# DLTs = [np.random.binomial(N, p) for N, p in zip(Ns, scenario)]
Ns = [3]
DLTs = [0]
scenarios = env.scenarios
num_scenarios = len(scenarios)
priors = [1.0/num_scenarios for _ in range(num_scenarios)]
likelihoods = [env.cal_likelihood(Ns=Ns, DLTs=DLTs, scenario=scenario) for scenario in scenarios]
posteriors = np.array([likelihood*prior for likelihood, prior in zip(likelihoods, priors)])
posteriors = posteriors/sum(posteriors)


# env.cal_MTD_posterior()
MTD_posterior = [0 for _ in range(env.D+1)]
for index in range(len(env.scenarios)):
    MTD_posterior[env.MTD_trues[index]] += posteriors[index]


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(np.arange(20), posteriors)
# plt.show()
fig2, ax2 = plt.subplots()
ax2.plot(np.arange(7), MTD_posterior)
plt.show()
print(rand_index)
print(MTD_true)
# %%
sum(posteriors)
# %%
