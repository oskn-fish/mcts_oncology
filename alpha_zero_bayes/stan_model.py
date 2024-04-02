# stan_model = """data {
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
#     DLT[i] ~ binomial(N_cohort, p[i]);
#   }
# }
# generated quantities {
#   int<lower=0, upper=N_cohort> pred_DLT[D];
#   for (i in 1:D) {
#     pred_DLT[i] = binomial_rng(N_cohort, p[i]);  
#   }
# }"""
"""
↑偏りが大きすぎるためむり．zhouの論文の方法でサンプリングする

a. Select one of the J dose levels as the MTD with equal probabilities.
b. Sample M ∼ Beta ðmaxf J − j; 0:5g; 1Þ, where j denotes the selected dose level, and set an upper bound B = φ +
(1 − φ) × M for the toxicity probabilities.
c. Repeatedly sample J toxicity probabilities uniformly on [0, B] until these correspond to a scenario in which dose
level j is the MTD.
"""


# 

# stan_model = """
# data {
#   int<lower=0> D;
#   int<lower=0> data_size;
#   int<lower=0> N_cohort;
#   real<lower=0.0, upper=1.0> phi;
#   int<lower=0, upper=D> draw_dose[data_size];
#   int<lower=0, upper=N_cohort> DLT[data_size];
# }

# parameters {  
#   positive_ordered[D] p;
  
#   real<lower=0.0, upper=1> M;
  
  
# }

# transformed parameters {
#   real<lower=0.0, upper=1.0> B;
#   positive_ordered[D] p_M;
#   B = phi+(1-phi)*M;
#   for (i in 1:D) {
#     p_M[i] = B*p[i];
#   }
# }

# //model {
# //  for (i in 1:data_size) {
# //    target += binomial_lpmf(N_cohort, p_M[i]);
# //  }

# model {
#   // p_Mは一様分布を縮尺したもの．一様分布は明示的に定義する必要ない
#   //target += uniform_lpdf(p|0.0, 1.0);
#   //for (i in 1:D) {
#   //  target += uniform_lpdf()
#   //}
#   //p_M ~ uniform(0.0, B);
#   p_M ~ uniform(0.0, M);
  
#   // 尤度（モデル）部分
#   for (i in 1:data_size) {
#     target += binomial_lpmf(DLT[i] | N_cohort, p_M[draw_dose[i]]);
#   }
  
  
  
  
#   // p_Mの上限BはMのアフィン変換
  
#   // Mの部分のtargetを追加
#   real lp[D];
#   for (i in 1:D) {
#     lp[i] = log(1/D) + beta_lpdf(M | fmax(D-i, 0.5), 1);
#   }
#   target += log_sum_exp(lp);
# }

# """

# pは一様分布だから設定いらない？（自動で推定される？）→よくわからん

# stan_model = """
# data {
#   int<lower=0> D;
#   int<lower=0> data_size;
#   int<lower=0> N_cohort;
#   real<lower=0.0, upper=1.0> phi;
#   int<lower=0, upper=D> draw_dose[data_size];
#   int<lower=0, upper=N_cohort> DLT[data_size];
# }

# parameters {  
#   ordered[D] p;
#   real<lower=0.0, upper=1.0> M;
# }

# transformed parameters {
#   real<lower=phi, upper=1.0> B;
#   ordered[D] p_B;
#   B = phi+(1-phi)*M;
#   for (i in 1:D) {
#     p_B[i] = B*inv_logit(p[i]);
#   }
# }

# model {
#   // p_Mは一様分布を縮尺したもの．一様分布は明示的に定義する必要ない
#   //target += uniform_lpdf(p|0.0, 1.0);
#   //for (i in 1:D) {
#   //  target += uniform_lpdf()
#   //}
#   //p_M ~ uniform(0.0, B);
#   //p_M ~ uniform(0.0, M);
  
#   // 尤度（モデル）部分
#   for (i in 1:data_size) {
#     target += binomial_lpmf(DLT[i] | N_cohort, p_B[draw_dose[i]]);
#   }
  
#   //for (i in 1:data_size) {
#   //  DLT[i] ~ binomial(N_cohort, p_B[draw_dose[i]]);
#   //  //DLT[i] ~ binomial(N_cohort, 0.5);
#   //}
  
  
  
  
#   // p_Mの上限BはMのアフィン変換
  
#   // Mの部分のtargetを追加
#   real lp[D];
#   for (i in 1:D) {
#     lp[i] = log(1.0/D) + beta_lpdf(M | fmax(D-i, 0.5), 1);
#     //lp[i] = log(1.0/D) + beta_lpdf(M | 0.5, 1);
#   }
#   target += log_sum_exp(lp);
# }

# """

# ↑よくわからんが動かんので，p ~ uniformではなく，beta分布に従うとしておく
# orderedがあまりに大きな値を出すので，inv_logitはほぼ1になってしまっている．

# stan_model = """

# parameters {
#   vector<lower=0.0, upper=1.0>[6] p_x;
# }

# transformed parameters {
#   vector<lower=0.0, upper=1.0>[6] p_y;
#   p_y = sort_asc(p_x);
# }

# model {
#   for (i in 1:6) {
#     p_x[i] ~ uniform(0.0, 1.0);
#   }
  
# }

# """

# ↑これはいい感じにuniformにsortされた分布になっているぽい

# stan_model = """
# data {
#   int<lower=0> D;
#   int<lower=0> data_size;
#   int<lower=0> N_cohort;
#   real<lower=0.0, upper=1.0> phi;
#   int<lower=0, upper=D> draw_doses[data_size];
#   int<lower=0, upper=N_cohort> draw_DLTs[data_size];
# }

# parameters {  
#   real<lower=0.0, upper=1.0> M;
#   vector<lower=0.0, upper=1.0>[D] p;
  
# }

# transformed parameters {
#   real<lower=phi, upper=1.0> B;
#   B = phi+(1-phi)*M;
  
#   vector<lower=0.0, upper=1.0>[D] p_ordered;
#   p_ordered = sort_asc(p);
#   vector<lower=0.0, upper=1>[D] p_B;
#   for (i in 1:D) {
#     p_B[i] = B*p_ordered[i];
#   }
# }

# model {
  
#   // 尤度（モデル）部分
#   for (i in 1:data_size) {
#     //target += binomial_lpmf(DLT[i] | N_cohort, p_B[draw_doses[i]]);
#     draw_DLTs[i] ~ binomial(N_cohort, p_B[draw_doses[i]]);
#   }
  
#   // p_Mの上限BはMのアフィン変換
  
#   // Mの部分のtargetを追加
#   real lp[D];
#   for (i in 1:D) {
#     lp[i] = log(1.0/D) + beta_lpdf(M | fmax(D-i, 0.5), 1);
#   }
#   target += log_sum_exp(lp);
  
#    // pの部分
#   for (i in 1:D) {
#     p[i] ~ uniform(0.0, 1.0);
#   }
# }

 
# """

stan_model = """
data {
  int<lower=0> D;
  int<lower=0> N_total;
  real<lower=0.0, upper=1.0> phi;
  int<lower=0, upper=N_total> draw_patients[D];
  int<lower=0, upper=N_total> draw_DLTs[D]; // too loose value range constraint. might cause errors 
}

parameters {  
  real<lower=0.0, upper=1.0> M;
  vector<lower=0.0, upper=1.0>[D] p;
  
}

transformed parameters {
  real<lower=phi, upper=1.0> B;
  B = phi+(1-phi)*M;
  
  vector<lower=0.0, upper=1.0>[D] p_ordered;
  p_ordered = sort_asc(p);
  vector<lower=0.0, upper=1>[D] p_B;
  for (i in 1:D) {
    p_B[i] = B*p_ordered[i];
  }
}

// 混合分布の取り扱い（混合モデルの取り扱い）https://kento1109.hatenablog.com/entry/2018/04/28/121855

// 混合モデル
// クラスタiのモデルのパラメータが分布p(s_i)に従うとき，
// データx_jのpdfは以下．
// \sum_i p(s_i)p(x_j | s_i)
// したがって，データx_1,,,x_nのデータの尤度は，
// \prod_j \sum_i p(s_i)p(x_j | s_i)
// logを取ると，
// \sum_j( log(\sum_i log(s_i)+log(p(x_j | s_i)) ))

// 今回はデータがMの一つだけ．
// log(\sum_i log(s_i)+log(M | s_i))

model {
  
  // 尤度（モデル）部分
  for (i in 1:D) {
    draw_DLTs[i] ~ binomial(draw_patients[i], p_B[i]);
  }
  
  // p_Mの上限BはMのアフィン変換
  
  // Mの部分のtargetを追加
  real lp[D];
  for (i in 1:D) {
    lp[i] = log(1.0/D) + beta_lpdf(M | fmax(D-i, 0.5), 1);
  }
  target += log_sum_exp(lp);
  
   // pの部分
  for (i in 1:D) {
    p[i] ~ uniform(0.0, 1.0);
  }
}

 
"""