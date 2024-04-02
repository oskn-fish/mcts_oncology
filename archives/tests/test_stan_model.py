"""
conda環境では動かないことに注意！

ValueError: The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc. となる．

venvで実行すること．
"""

#%%
import pandas as pd
import numpy as np
def generate_data(config):
    scenarios = np.array([
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
            ])
    data = []
    scenario_idx = np.random.randint(low = 0, high = 20)
    p_true = scenarios[scenario_idx]
    MTD_true = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
    MTD_true = MTD_true[scenario_idx]
    
    # observation関連の変数
    Ns   = np.zeros(config["D"])
    
    trial = 0
    while sum(Ns) < config["N_total"]:
        # とりあえず0x4, 1x4, 2x4回とする
        draw_dose = trial//4
        draw_DLT  = np.random.binomial(n = config["N_cohort"], p = p_true[draw_dose])
        Ns[draw_dose] += config["N_cohort"]
        # doseは1から始まる（stan, paper）ことに対応
        data.append({"draw_dose":draw_dose+1, "DLT": draw_DLT})
        trial += 1
    return data, p_true
            
# "D"はDoseのこと．今回は6種類の用量．
config = {"D":6, "N_cohort":3, "N_total":36}
data, p_true = generate_data(config)
data = pd.DataFrame(data)
data = data.to_dict(orient="list")
data_size = len(data["DLT"])
data["D"] = 6   
data["data_size"] = data_size
data["N_cohort"] = 3
data["phi"] = 0.25
# data["p_lower"] = 0.
# data["p_upper"] = 1
# data["pred_dose"] = [3,4]
# data["p_true"] = p_true
#%%
print(data)

#%%
import stan
import nest_asyncio
nest_asyncio.apply()

# stan_code = """
# data {
#   int<lower=0> D;
#   int<lower=0> data_size;
#   int<lower=0> N_cohort;
#   real<lower=0> p_lower;
#   real<lower=0> p_upper;
#   int<lower=0, upper=D> draw_dose[data_size];
#   int<lower=0, upper=N_cohort> DLT[data_size];
# }
# parameters {
#   real<lower=p_lower, upper=p_upper> p_1;
#   real<lower=p_1, upper=p_upper> p_2;
#   real<lower=p_2, upper=p_upper> p_3;
#   real<lower=p_3, upper=p_upper> p_4;
#   real<lower=p_4, upper=p_upper> p_5;
#   real<lower=p_5, upper=p_upper> p_6;
# }
# transformed parameters {
#   real p[D];
#   p[1] = p_1;
#   p[2] = p_2;
#   p[3] = p_3;
#   p[4] = p_4;
#   p[5] = p_5;
#   p[6] = p_6;
# }
# model {
#   p[1] ~ uniform(p_lower, p_upper);
#   for (i in 2:D) {
#     p[i] ~ uniform(p[i-1], p_upper);
#   }
#   for (i in 1:data_size){
#     DLT[i] ~ binomial(N_cohort, p[draw_dose[i]]);
#   }
# }
# generated quantities {
#   int<lower=0, upper=N_cohort> pred_DLT[D];
#   for (i in 1:D) {
#     pred_DLT[i] = binomial_rng(N_cohort, p[i]);  
#   }
# }
# """

# 生成したデータをcropして，部分的データ作成
# import copy
# import arviz as az
# import matplotlib.pyplot as plt
# def data_crop(data: dict, size: int) -> None:
#   cropped = copy.deepcopy(data)
#   cropped["draw_dose"] = cropped["draw_dose"][0:size]
#   cropped["DLT"] = cropped["DLT"][0:size]
#   cropped["data_size"] = size
#   if len(cropped["DLT"])!=cropped["data_size"]:
#     raise SyntaxError(f'len(cropped["DLT"]):{len(cropped["DLT"])}, whereas cropped["data_size"]:{cropped["data_size"]}')
#   return cropped
# #%%
# # データを追加しながら事後分布作成
# for i in range(1,13):
#   cropped_data = data_crop(data, i)
#   posterior = stan.build(stan_code, data=cropped_data, random_seed=1)

#   fit = posterior.sample(num_chains=4, num_samples=1000)
#   df = fit.to_frame()

#   #%%
#   p
#   # plt.show()
#   plt.savefig(f"img/posterior(samplesize:{i}).png")
#   #%%
#   print(data["p_true"])
#   plt.plot(range(6),data["p_true"])

#   #%%
#   # pred_p = df.median()[7:13].to_list()
#   # plt.plot(range(1,7), pred_p)
#   df.describe(percentiles=[0.025, 0.975]).loc[["2.5%","mean","97.5%"],["p_1","p_2","p_3","p_4","p_5","p_6"]].transpose().plot()
#   plt.savefig(f"img/predivtive(samplesize:{i}).png")

from alpha_zero_bayes.stan_model import stan_model
posterior = stan.build(stan_model, data=data, random_seed=1)

fit = posterior.sample(num_chains=4, num_samples=1000)
df = fit.to_frame()
# %%
import arviz as az
az.plot_trace(fit)
# %%
df.describe()
# %%
df[["p_B.1", "p_B.2", "p_B.3", "p_B.4", "p_B.5", "p_B.6"]].sample(50).values
#%%
import matplotlib.pyplot as plt
ndarray_list = df[["p_B.1", "p_B.2", "p_B.3", "p_B.4", "p_B.5", "p_B.6"]].sample(50).values
for i in range(50):
    plt.plot(np.arange(6), ndarray_list[i])
plt.show()
    


# %%
