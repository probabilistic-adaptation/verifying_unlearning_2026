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
            "num_epochs": 75,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_print_freq": 5,
            },
        
        "unlearning": {
            "learning_rate": {
                "GA": 5e-5, # recall we are now doing SGD, so the learning rate is different from Adam
                "FT": 5e-3, # 1e-2 sometimes causes calamitous weight divergence (at least on CIFAR10)
                "NegGrad_plus": 1e-4,
                "RL": 1e-4,
                "boundary_shrink": 1e-5, # from original paper
                "bad_teacher": 5e-5,
                "scrub": 5e-4 # 5e-4 is actually what they use for one "round"
                },
            "batch_print_freq": {
                "GA": 1,
                "FT": 8,
                "NegGrad_plus": 8,
                "RL": 8,
                "boundary_shrink": 1,
                "bad_teacher": 3,
                "scrub": 3,
                }
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
                "pretrained_seed": 1, # 4 is real 
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
            "unlearning": {
                "learning_rate": {
                    "FT": 1e-3,
                }
            }
            }
            )
    },
        
    
    "Lacuna10": {},
    

    
    
   
}