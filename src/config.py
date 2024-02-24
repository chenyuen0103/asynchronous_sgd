config = {
    "n_data": 10000,
    "dim": 400,
    "noise_scale": 1e-5,
    "batch_size": 256,  # This is used differently now, ensure to adjust its use in run_training
    "num_workers": 4,
    "lr": 0.01,
    "lr_decay": 0,
    "iterations": 200,
    "asynchronous": True,
    "it_check": 20,
    "seed": 42,
    "delay_adaptive": False,  # Added based on run function arguments
    "one_bad_worker": False,   # Added based on run function arguments
    # "max_seed" is not explicitly in the run function arguments, but could be necessary for seed generation
    "max_seed": 1000000,       # An arbitrary large number for seed generation
    # Add any other configurations as needed
}
