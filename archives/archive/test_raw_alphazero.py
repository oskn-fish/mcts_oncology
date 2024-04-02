from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.rllib.examples.env.cartpole_sparse_rewards import CartPoleSparseRewards
from ray.rllib.algorithms.alpha_zero.models.custom_torch_models import DenseModel
config = AlphaZeroConfig()   
config = config.training(sgd_minibatch_size=256,model={"custom_model": DenseModel})   
config = config.resources(num_gpus=4)   
config = config.rollouts(num_rollout_workers=0)   
config = config.environment(env=CartPoleSparseRewards)
print(config.to_dict()) 
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()  
for i in range(100):
    
    result = algo.train() 
    print(result)