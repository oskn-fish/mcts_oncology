import stan
import nest_asyncio
import pandas as pd
import numpy as np
from  stan_model import model
nest_asyncio.apply()

NUM_CHAIN = 4
NUM_SAMPLES = 1000
D = 6
N_COHORT = 3
P_LOWER = 0.0
P_UPPER = 1.0
P_TRUE = np.array([0.35 , 0.522, 0.688, 0.8  , 0.8  , 0.8  ])

class PredictiveModel():
    def __init__(self, data: dict[str, int]):
        """
        {'draw_dose': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 'DLT': [1, 1, 1, 1, 1, 0, 1, 0, 3, 2, 1, 0], 'D': 6, 'data_size': 12, 'N_cohort': 3, 'p_lower': 0.0, 'p_upper': 1)}
        """
        # TODO change this initial value to be class initial argument
        self.all_data = {"draw_dose": [], "DLT": [], "D": D, "data_size": 0, "N_cohort": N_COHORT, "p_lower": P_LOWER, "p_upper": P_UPPER}
        # path指定がめんどくさいので．モジュールにする
        # with open("model.stan") as f: 
        #     self.model = f.read()
        # self.update()
        self.update_mcmc_samples(data)
        
        
    def sample_data(self, dose: int) -> int:
        return np.random.choice(self.df["pred_DLT."+str(dose)].values)
    
    # @property
    def get_mcmc_samples(self) -> pd.DataFrame:
        # TODO optimize observation definition
        # obs = self.df.loc[:,["p_1", "p_2", "p_3", "p_4", "p_5", "p_6"]].describe().loc[["mean", "std"]]
        # return obs
        return self.df
    
    def update_mcmc_samples(self, data: dict[str, int]):
        self._add_data(data)
        # TODO: baysian update
        self.posterior = stan.build(model, data=self.all_data, random_seed=1)
        self.fit = self.posterior.sample(num_chains=NUM_CHAIN, num_samples=NUM_SAMPLES)
        self.df = self.fit.to_frame()
       
    def _add_data(self, data: dict[str, int]):
        self.all_data["draw_dose"].append(data["draw_dose"])
        self.all_data["DLT"].append(data["DLT"])
        self.all_data["data_size"] += 1
        
        
if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    data = {"draw_dose":0, "DLT":0}
    predictive_model = PredictiveModel(data)
    az.plot_trace(predictive_model.fit)
    plt.show()
    print("data 1")
    for i in range(1,7):
        sample = predictive_model.sample_data(i)
        print(sample)

    az.plot_trace(predictive_model.fit)
    plt.show()
    print("data 2")    
    data = {"draw_dose":1, "DLT":1}
    predictive_model.update_mcmc_samples(data)
    for i in range(1,7):
        sample = predictive_model.sample_data(i)
        print(sample,end="\t")
    
    az.plot_trace(predictive_model.fit)
    plt.show()
    print("data 3")        
    data = {"draw_dose":2, "DLT":3}
    for i in range(1,7):
        sample = predictive_model.sample_data(i)
        print(sample,end="\t")
    