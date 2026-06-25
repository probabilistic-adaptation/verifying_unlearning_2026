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
                "num_epochs": 10

            },
            "GA": {
                "print_freq": 1,

                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10
            },
            "NegGrad_plus": {
                "print_freq": 8,
                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10,
                "alpha": 0.95 # taken from Kurmanji et al 2023
                },

            "RL": {
                "print_freq": 8,
                # Golatkar et al, 2020a
                "lr": .01,
                "opt": "SGD",
                "weight_decay": 5e-4,
                "num_epochs": 10
                },
            
            "SalUn": {},

            "boundary_shrink": {
                "print_freq": 1,
                # Fan et al 2024
                "lr": 1e-5,
                "num_epochs": 10,
                "bound": 0.1,
            },
            
            "boundary_expansion" :{
                
                # Fan et al 2024
                "lr": 1e-5, # original paper uses 1e-5, Fan et al lists 10^-5 which i dont think is the same
                "num_epochs": 10,
            },

            "bad_teacher": {
                "print_freq": 3,
                
                # chundawat et al 2023 (original paper)
                "lr": 1e-4, 
                "num_epochs": 2,
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
                "lr_decay_by": .1,
                "sgda_epochs": 3,
                "maxsteps": 2
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
                    "class": 5,
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