#%%
# import stan
# import nest_asyncio
# nest_asyncio.apply()
# import arviz as az
# import matplotlib.pyplot as plt

# from alpha_zero_bayes.stan_prior import stan_prior

# data = {"D":6}

# posterior = stan.build(stan_prior, data=data, random_seed=1)
# fit = posterior.sample(num_chains=4, num_samples=1000)
# df = fit.to_frame()
# az.plot_trace(fit)
# plt.show()
# # %%
# print(df[["p_1", "p_2"]].values)
# %%

# そもそも，posterior計算するのでなければ，stan使わずnumpyで乱数せいせいすればよい．
import numpy as np
import matplotlib.pyplot as plt

D = 6
φ = 0.25
def generate_p_true():
    MTD = np.random.randint(D)
    M = np.random.beta(max(D-MTD, 0.5), 1.0)
    
    B = φ + (1-φ)*M

    done = False
    while not done:
        p_candidate = B*np.random.rand(D)
        p_candidate.sort()
        # MTD should be the closest dose to DLT = φ
        # matsuura, zhouにならう．
        p_difference = np.abs(p_candidate-φ)
        closest_dose = p_difference.argmin()
        if closest_dose==MTD:
            if not (closest_dose==0 and p_difference[0]>0.1):
                done = True
    p_true = p_candidate
    return {"MTD": MTD, "p_true": p_true, "M":M}
    
#%%

# for i in range(50):
    # plt.plot(np.arange(D), generate_p_true()["p_true"])
m_list = []
for i in range(10):
    m_list.append(generate_p_true()["M"])
print(m_list)
plt.show()

# %%
