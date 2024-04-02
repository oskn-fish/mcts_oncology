# stan_prior = """
# parameters {
#     real<lower=0.0, upper=1.0> p_1;
#     real<lower=p_1, upper=1.0> p_2;
#     real<lower=p_2, upper=1.0> p_3;
#     real<lower=p_3, upper=1.0> p_4;
#     real<lower=p_4, upper=1.0> p_5;
#     real<lower=p_5, upper=1.0> p_6;
# }

# model {
#     p_1 ~ uniform(0.0, 1.0);
#     p_2 ~ uniform(p_1, 1.0);
#     p_3 ~ uniform(p_2, 1.0);
#     p_4 ~ uniform(p_3, 1.0);
#     p_5 ~ uniform(p_4, 1.0);
#     p_6 ~ uniform(p_5, 1.0);
# }
# """



# stan_prior = """
# parameters {
#     real<lower=0.0, upper=1.0> p_6;
#     real<lower=0.0, upper=p_6> p_5;
#     real<lower=0.0, upper=p_5> p_4;
#     real<lower=0.0, upper=p_4> p_3;
#     real<lower=0.0, upper=p_3> p_2;
#     real<lower=0.0, upper=p_2> p_1;
# }

# model {
#     p_6 ~ uniform(0.0, 1.0);
#     p_5 ~ uniform(0.0, p_6);
#     p_4 ~ uniform(0.0, p_5);
#     p_3 ~ uniform(0.0, p_4);
#     p_2 ~ uniform(0.0, p_3);
#     p_1 ~ uniform(0.0, p_2);
# }
# """

# ↑ phi=0.25より大幅に大きかったり，小さかったりする分布になってしまう．
# Comparative review of novel model‐assisted designs for phase I clinical trials を参考に構成．

# We generated true dose‐toxicity scenarios using the pseudo‐uniform algorithm proposed by Clertant and O'Quigley.26 Given a target DLT probability φ and J dose levels, we generated scenarios as follows:
# a. Select one of the J dose levels as the MTD with equal probabilities.
# b. Sample M ∼ Beta (max{ J − j; 0.5},1), where j denotes the selected dose level, and set an upper bound B = φ + (1 − φ) × M for the toxicity probabilities. → Mは上限を決めるだけ．
# c. Repeatedly sample J toxicity probabilities uniformly on [0, B] until these correspond to a scenario in which dose
# level j is the MTD.

stan_prior = """
data {
    int<lower=1> D;
}
parameters {
    int<lower=1, upper=D>dose;
    real p[D];
    real M;
}
model {
    dose ~ uniform(1, D);
    M ~ beta(max([D-i, 0.5]),1);
    for (i in 1:D)
        p[i] ~ uniform(0, M);
}
"""