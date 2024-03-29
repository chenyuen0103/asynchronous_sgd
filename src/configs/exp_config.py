config = {
    "n_data": 10,
    "dim": 4,
    "noise_scale": 1e-5,
    "batch_size": 5,  # This is used differently now, ensure to adjust its use in run_training
    "num_workers": 2,
    "lr": 0.01,
    "lr_decay": 0,
    "iterations": 200,
    "asynchronous": True,
    "it_check": 100,
    "seed": 42,
    "delay_adaptive": False,  # Added based on run function arguments
    "one_bad_worker": False,   # Added based on run function arguments
    # "max_seed" is not explicitly in the run function arguments, but could be necessary for seed generation
    "max_seed": 1000000,
    "cost_aware": False,# An arbitrary large number for seed generation
    # Add any other configurations as needed
    'COST_DISTRIBUTION_PARAMS':{
    "communication": {
        "distribution": "lognormal",
        "mu_m": 0,  # Mean of the log-normal distribution for communication
        "sigma_m": 1e-6,  # Standard deviation of the log-normal distribution for communication
    },
    "computation": {
        "distribution": "lognormal",
        "mu": 0,  # Mean of the log-normal distribution for computation
        "sigma": 1e-6,  # Standard deviation of the log-normal distribution for computation
    },
    # "n_workers": 4,  # Number of workers in the distributed system
    }
}
