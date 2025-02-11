
#%%
import matplotlib.pyplot as plt
array = [
                [0.25,0.35,0.465,0.584,0.694,0.786],
                [0.171,0.25,0.35,0.465,0.584,0.694],
                [0.113,0.171,0.25,0.35,0.465,0.584],
                [0.073,0.113,0.171,0.25,0.35,0.465],
                [0.05,0.073,0.113,0.171,0.25,0.35],
                [0.05,0.05,0.073,0.113,0.171,0.25],
                [0.35,0.465,0.584,0.694,0.786,0.8],
                [0.21,0.35,0.522,0.688,0.8,0.8],
                [0.116,0.21,0.35,0.522,0.688,0.8],
                [0.061,0.116,0.21,0.35,0.522,0.688],
                [0.05,0.061,0.116,0.21,0.35,0.522],
                [0.05,0.05,0.061,0.116,0.21,0.35],
                [0.05,0.05,0.05,0.061,0.116,0.21],
                [0.35,0.522,0.688,0.8,0.8,0.8],
                [0.29,0.486,0.686,0.8,0.8,0.8],
                [0.15,0.29,0.486,0.686,0.8,0.8],
                [0.071,0.15,0.29,0.486,0.686,0.8],
                [0.05,0.071,0.15,0.29,0.486,0.686],
                [0.05,0.05,0.071,0.15,0.29,0.486],
                [0.05,0.05,0.05,0.071,0.15,0.29]
            ]

# %%
import japanize_matplotlib
x = range(1,7)
fig, ax = plt.subplots()
for i in range(len(array)):
    ax.plot(x, array[i])
# plt.savefig("img/matsuura_visualize.png")
ax.set_xlabel("用量水準", fontsize=16)
ax.set_ylabel("用量制限毒性発現確率", fontsize=16)
plt.show()
# %%
