import copy

def deep_update(base, overrides):
    base = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base

base = {
    
    "ResNet":{
        "training": {
            "num_epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_print_freq": 5,
            },
        
        "unlearning": {
            "FT": {
                "print_freq": 8,

                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10,
                "save_checkpoints_at": [5, 10],

            },
            "GA": {
                "print_freq": 1,

                # Jia et al 2024
                "lr": 2e-4, # anything higher than this results in calamitous model utility, any lower (1e-4 from jia et al 2024) and the model doesnt unlearn anything within 5 epochs
                "opt": "SGD",
                "weight_decay": 5e-4, # this is actually not from the paper, we borrow this in keeping with other weight decay applied in Golatkar et al 2020a
                "num_epochs": 5,
                "save_checkpoints_at": [3, 5],
            },
            "NegGrad_plus": {
                "print_freq": 8,
                
                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10,
                "save_checkpoints_at": [5, 10],
                "alpha": 0.999 # Kurmanji et al 2023 use 0.95, but that causes calamitous model collapse
                },

            "RL": {
                "print_freq": 8,
                
                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10,
                "save_checkpoints_at": [5, 10],
                },
            
            "SalUn": {},

            "boundary_shrink": {
                "print_freq": 1,
                
                # Fan et al 2024
                "lr": 1e-5,
                "num_epochs": 10,
                "save_checkpoints_at": [5, 10],
                "bound": 0.1,
                "opt": "SGD",
            },
            
            "boundary_expanding" :{
                
                # Fan et al 2024
                "lr": 1e-5, # original paper uses 1e-5, Fan et al lists 10^-5 which i dont think is the same
                "num_epochs": 10,
                "save_checkpoints_at": [5, 10],
                "opt": "SGD",
            },

            "bad_teacher": {
                "print_freq": 3,
                
                # chundawat et al 2023 (original paper)
                "lr": 1e-4, 
                "num_epochs": 2,
                "save_checkpoints_at": [1, 2],
                "opt": "Adam",

                # Kurmanji et al 2023
                # whole retain set, instead of 30% of retain set from original paper
                
                
            },
            "scrub": {
                "print_freq": 3,

                # Kurmanji et al 2023,
                "opt": "Adam",
                "weight_decay": 5e-4,
                "lr": 5e-4,
                # "lr_decay_by": .1, we're going to try cosine annealing, for consistency with our pretraining and retrain from scratch protocol
                "num_epochs": 3,
                "maxsteps": 2,
                "save_checkpoints_at": [1, 2, 3],
            },
            }
        }
    }


hyperparams = {
    
    
    "CIFAR10": {

        "num_classes": 10,
        "batch_size": 512,
        "num_workers": 4,
        "items_to_unlearn": {
            "class": 5,
            "percents": .2,
        },
        

        "ResNet": deep_update(base['ResNet'], {
            "training": {
                "pretrained_seed": 4, # 4 is real 
                "retrained_from_scratch_seeds": {
                    "class": 4,
                    "percents": -99
                    }
                }
            })

    },
    "SVHN": {
        
        "num_classes": 10,
        "batch_size": 512,
        "num_workers": 4,
        "class_to_unlearn": 5,
        "percents_to_unlearn": .2,

        "ResNet": deep_update(base['ResNet'], {
            "training": {
                "pretrained_seed": -99, 
                "retrained_from_scratch_seeds": {
                    "class": -99
                    }
                },
            # "unlearning": {
            #     "learning_rate": {
            #         "FT": 1e-3,
            #     }
            # }
            }
            )
    },
        
    
    "Lacuna10": {},
    

    
    
   
}