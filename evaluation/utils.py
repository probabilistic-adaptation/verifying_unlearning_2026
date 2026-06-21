import torch.nn as nn

# forget, retain, and test accuracy + entropy + m_entropy
from trainer.val import validate

# MIA (threshold function + SVC)
from evaluation.MIA import MIA
from evaluation.SVC_MIA import SVC_MIA

from evaluation.distances import l2_distance, JSDiv, ZRFScore, absolute_distance, kl_div_metric
from scipy.stats import wasserstein_distance


def measure_unlearning_metrics(model, dataloaders, device, base_out = None, retrained_out = None):

    # fill in base information
    criterion= nn.CrossEntropyLoss()
    model.eval() # just overwhelmingly confirm that we are in eval mode here, regardless of whether the model was passed in eval mode
    
    # add in metrics, as you'd like
    # we deliberately set `print_freq` very high so we dont actually print anything
    print("Evaluating forget set metrics...\n")
    forget_out = validate(dataloaders["forget"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)

    print("Evaluating retain set metrics...\n")
    retain_out = validate(dataloaders["retain"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)

    print("Evaluating test set metrics...\n")
    test_out = validate(dataloaders["test"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)

    results = {
        "acc": {
            "forget":forget_out["avg_acc"],
            "retain":retain_out["avg_acc"],
            "test": test_out["avg_acc"],
            },
        "loss": {
            "forget":forget_out["avg_loss"],
            "retain":retain_out["avg_loss"],
            "test": test_out["avg_loss"],
            },
        "entropy": {
            "forget":forget_out["avg_entr"],
            "retain":retain_out["avg_entr"],
            "test": test_out["avg_entr"],
            },
        "m_entropy": {
            "forget":forget_out["avg_m_entr"],
            "retain":retain_out["avg_m_entr"],
            "test": test_out["avg_m_entr"],
            }
            }

    # now need to gather output dists and labels for retain_one and retain_two, for the MIAs
    # Both MIAs recompute entropy and m-entropy, which is a bit of wasted computation,
    # could refactor that later
    # at least this way, they do not have to pass through the data loader for forget and test again
    print("Running through `retain_one` and `retain_two` for MIAs...\n")
    retain_one_out = validate(dataloaders["retain_one"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    retain_two_out = validate(dataloaders["retain_two"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)


    # gathering outputs for unlearned model
    unlearned_out = {
        "forget": forget_out,
        "retain": retain_out,
        "test": test_out,
        "retain_one": retain_one_out,
        "retain_two": retain_two_out,
    }
    
    print("Performing threshold MIA attack...\n")
    attack_results = MIA(
        shadow_train_inputs = (retain_one_out["probs"].numpy(), retain_one_out["targets"].numpy()),
        shadow_test_inputs = (test_out["probs"].numpy(), test_out["targets"].numpy()),
        target_train_inputs = (retain_two_out["probs"].numpy(), retain_two_out["targets"].numpy()),
        target_test_inputs = (forget_out["probs"].numpy(), forget_out["targets"].numpy()),
        model = model,
        device = device
        )
    
    # attack results should look something like this:
    # attack_results = {
    #     "correctness": #,
    #     "class": {
    #         "confidence": #,
    #         "entropy": #,
    #         "m_entropy": #
    #     },
    #     "no_class": {
    #         "confidence": #,
    #         "entropy": #,
    #         "m_entropy": #
    #     }
    # }
    
    results["threshold_MIA"] = attack_results

    # # COMMENTING OUT SVC FOR NOW, TAKES TOO LONG

    # print("Performing SVC MIA attack...\n")
    # # `device` is inferred by whatever the device the model is on - dont explicitly pass it (might be smart to do this in other areas too, so you dont have to carry `device` around)
    # # need the (probs, labels) outputs for each data loader we care about - some have already been computed through other passes through forget and test

    # # we do not need to convert this output using ".numpy()", since the SVC operations are done using torch
    # attack_results = SVC_MIA(
    #     shadow_train_inputs = (retain_one_probs, retain_one_labels), 
    #     shadow_test_inputs = (test_probs, test_labels),
    #     target_train_inputs = (retain_two_probs, retain_two_labels),
    #     target_test_inputs = (forget_probs, forget_labels),
    #     model = model
    #     )
    # results["SVC_MIA"] = attack_results



    print("Evaluating differences/distances between unlearned and retrained outputs ...\n")
    print("Absolute Distance ...")
    dist1 = absolute_distance( 
        unlearned_out['forget']['probs'], 
        retrained_out['forget']['probs']
        )
    print("L2 ...")
    dist2 = l2_distance( 
        unlearned_out['forget']['probs'], 
        retrained_out['forget']['probs']
        )
    print("JS Divergence ...")
    dist3 = JSDiv( 
        unlearned_out['forget']['probs'], 
        retrained_out['forget']['probs']
        )
    print("KL Divergence - retrained vs. unlearned, avg over forget + retain ...")
    dist4 = kl_div_metric( 
        retrained_out['forget']['probs'],
        unlearned_out['forget']['probs'],
        retrained_out['retain']['probs'],
        unlearned_out['retain']['probs']
        
        )
    print("Wasserstein distance - unlearned, forget vs. test losses ...")
    dist5 = wasserstein_distance(
        unlearned_out['forget']["losses"], 
        unlearned_out['test']["losses"]
        )
    
    results.update({ 
        
        "outputs": {
            "forget": {
                "retrained_vs_unlearned": {
                    "absolute_distance": dist1,
                    "l2_distance": dist2,
                    "JS_divergence": dist3,
                    }
                },
            "forget_test_avg": {
                "retrained_vs_unlearned": {
                    "KL_divergence": dist4
                    }
                },
            "forget_vs_test": {
                "unlearned": {
                    "wasserstein_distance": dist5
                    }
                }
            }
        })


    return results, unlearned_out
    

    


