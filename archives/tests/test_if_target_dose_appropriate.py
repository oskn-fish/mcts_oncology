#%%
import stan
from alpha_zero_bayes.stan_prior import stan_prior
import nest_asyncio
nest_asyncio.apply()

posterior = stan.build(stan_prior, random_seed=1)
fit = posterior.sample(num_chains=4, num_samples=1000)
df = fit.to_frame()[["p_1", "p_2", "p_3", "p_4", "p_5", "p_6"]]
random_p_true = df.sample().values[0]
p_true = random_p_true
# %%
import arviz as az
az.plot_trace(fit)
# %%
