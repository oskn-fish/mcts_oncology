import rpy2
import sys
import os
from rpy2 import robjects 
r = robjects.r
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RLE.env.RLEEnv_expectation_negative_reward_penalty_continue_evaluate import RLEEnv_expectation_negative_reward_penalty_continue_evaluate
import numpy as np


scenario = 0
D = 6
N_cohort = 3
N_total = 36

env_config = {
    "D": D,
    "N_cohort": N_cohort,
    "N_total": N_total,
    "DLT_max": 1000,
    "penalty_abs": 0,
    "scenario": scenario
}
env = RLEEnv_expectation_negative_reward_penalty_continue_evaluate(env_config)

def obs_to_str(input_str, last_obs, new_obs):

    """
    obs = [current_dose/(self.D-1), self.Ns/self.N_total, self.DLTs/self.N_total, sum(self.DLTs)/self.N_total, sum(self.Ns)/self.N_total]
    
    """
    diff_obs = new_obs - last_obs
    diff_Ns = np.round(diff_obs[1:D+1]*N_total).astype(int)
    diff_DLTs = np.round(diff_obs[D+1:2*D+1]*N_total).astype(int)
    dose_index = np.where(diff_Ns==3)[0][0]
    DLTs = diff_DLTs[dose_index]
    dose_index_from_one = dose_index+1
    diff_str = " "+str(dose_index_from_one)+"N"*(N_cohort-DLTs)+"T"*DLTs
    return input_str+diff_str


r("skeleton <- c(0.062, 0.140, 0.25, 0.376, 0.502, 0.615)")
r("target <- 0.25")
r("model <- get_trialr_crm(skeleton=skeleton, target=target, model='empiric', beta_sd=sqrt(2))")


num_simulations = 10000
for iteration in range(num_simulations):
    last_observation = np.array([0]*(2*D+3))
    new_observation, info = env.reset()
    input_str = ""
    
    done = False
    while not done:
        input_str = obs_to_str(input_str,last_observation, new_observation)
        r("trial_result <- "+input_str)[0]
        r(r"fit <- model %>% fit(trial_result)")
        dose = r(r"fit %>% recommended_dose()")
        print(dose)
        done = True

    """

    # "2NNT"
    # dose_index = 2, two non toxicity patients, one toxicity patient
    # this continues like "1TTT 2NNT 3NNN 4"
    # you need to add these 4 length string after each trial(probably)

    skeleton <- c(0.062, 0.140, 0.25, 0.376, 0.502, 0.615)
    target <- 0.25

    #  get_trialr_crm
    model <- get_trialr_crm(skeleton=skeleton, target=target, model='empiric', beta_sd=sqrt(2))
    trial_result <- '1NNT' 2NNN'
    fit <- model %>% fit(trial_result)
    fit %>% recommended_dose()
    """
    

    
    
        
        

    """
    /home/chinen/R/lib/R/etc/ldpaths
    """
