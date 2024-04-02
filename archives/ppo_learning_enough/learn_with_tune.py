import random

import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining

if __name__ == "__main__":
    import argparse
    
    def explore(config):
        """
        train_batch_sizeはreplay bufferからとってくるサンプル
        sgd_mini_batchはsgdを回すデータサイズ
        
        """
        if config["train_batch_size"] < config["sgd_mini_batch_size"]*2:
            config["train_batch_size"] = config["sgd_mini_batch_size"]*2
            
        """
        number of epochs to execute per train batch 
        """
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
            
        return config
    
    # mutation = mutant
    
    """
    "puct_coefficient": 1.0,
    "num_simulations": 100,
    "temperature": 0.5,
    "dirichlet_epsilon": 0.25,
    "dirichlet_noise": 0.03,
    "argmax_tree_policy": False,
    "add_dirichlet_noise": True,
    """
    
    hyperparam_mutations = {
        "puct_coefficient": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "temperature": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "dirichlet_epsilon": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "dirichlet_noise": tune.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
    }
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration", # The training result attr to use for comparing time. Note that you can pass in something non-temporal such as training_iteration as a measure of progress, the only requirement is that the attribute should increase monotonically.
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )
