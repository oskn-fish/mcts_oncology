import stan
import numpy
# for jupyter
import nest_asyncio
nest_asyncio.apply()

def sample_posterior(data: dict, actinon: int):
    # data["draw_dose"] = list
    # data["DLT"] = list
    stan_code = """
        data {
            int<lower=0> D;
            int<lower=0> data_size;
            int<lower=0> N_cohort;
            real<lower=0> p_lower;
            real<lower=0> p_upper;
            int<lower=0, upper=D> draw_dose[data_size];
            int<lower=0, upper=N_cohort> DLT[data_size];
            int<lower=0, upper=D> pred_dose[3];
        }
        parameters {
            real<lower=p_lower, upper=p_upper> p_1;
            real<lower=p_1, upper=p_upper> p_2;
            real<lower=p_2, upper=p_upper> p_3;
            real<lower=p_3, upper=p_upper> p_4;
            real<lower=p_4, upper=p_upper> p_5;
            real<lower=p_5, upper=p_upper> p_6;
        }
        transformed parameters {
            real p[D];
            p[1] = p_1;
            p[2] = p_2;
            p[3] = p_3;
            p[4] = p_4;
            p[5] = p_5;
            p[6] = p_6;
        }
        model {
            p[1] ~ uniform(p_lower, p_upper);
            for (i in 2:D) {
                p[i] ~ uniform(p[i-1], p_upper);
        }
            for (i in 1:data_size){
                DLT[i] ~ binomial(N_cohort, p[draw_dose[i]]);
            }
        }
        generated quantities {
            int<lower=0, upper=N_cohort> pred_DLT[3];
            for (i in 1:3) {
                pred_DLT[i] = binomial_rng(N_cohort, p[pred_dose[i]]);  
            }
        }
    """
    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df_fit = fit.to_frame()
    
    # action: [0, 1, 2] = [down, stay, up]
    previous_action = max(data)
    
    if previous_action == 0:
        pass
    data["pred_dose"] = [previous_action-1, previous_action, previous_action+1]
#%% 
    