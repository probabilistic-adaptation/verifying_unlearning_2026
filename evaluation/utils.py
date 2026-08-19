import torch
import torch.nn as nn

# forget, retain, and test accuracy + entropy + m_entropy
from trainer.val import validate

# MIA (threshold function + SVC)
from evaluation.MIA import MIA, entropy_threshold_MIA
from evaluation.SVC_MIA import SVC_MIA

from evaluation.distances import l2_distance, JSDiv, absolute_distance, kl_div_metric, pred_distribution_difference, normalized_confusion_distance
from evaluation.weight_differences import weight_distance
from evaluation.residual_information import fisher_information_bound, information_score, efficacy, efficacy_upper_bound
from scipy.stats import wasserstein_distance, ks_2samp


def forget_retain_test_validation(model, dataloaders, device, criterion, compute_fisher = False):
    
    # we deliberately set `print_freq` very high so we dont actually print anything

    print("Evaluating forget set metrics...\n")
    forget_out = validate(dataloaders["forget"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False, compute_fisher = compute_fisher)

    print("Evaluating retain set metrics...\n")
    retain_out = validate(dataloaders["retain"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False, compute_fisher = compute_fisher)

    print("Evaluating test set metrics...\n")
    # as of now, no metrics require computing FIM over the test set, so we hard-code False
    test_out = validate(dataloaders["test"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False, compute_fisher = False)

    out = {
        "forget": forget_out,
        "retain": retain_out,
        "test": test_out
    }

    return out


def measure_solo_metrics(model, dataloaders, device, seed, compute_fisher = False):
    """
    Measure all the unlearning metrics which do NOT require a reference model for comparison
    """

    criterion= nn.CrossEntropyLoss()
    model.eval() # just overwhelmingly confirm that we are in eval mode here, regardless of whether the model was passed in eval mode
    
    # add in metrics, as you'd like
    main_out = forget_retain_test_validation(model, dataloaders, device, criterion, compute_fisher=compute_fisher)

    results = {
        "acc": {
            "forget": main_out['forget']["avg_acc"],
            "retain": main_out["retain"]["avg_acc"],
            "test": main_out["test"]["avg_acc"],
            },
        "loss": {
            "forget": main_out['forget']["avg_loss"],
            "retain": main_out["retain"]["avg_loss"],
            "test": main_out["test"]["avg_loss"],
            },
        "entropy": {
            "forget": main_out['forget']["avg_entr"],
            "retain": main_out["retain"]["avg_entr"],
            "test": main_out["test"]["avg_entr"],
            },
        "m_entropy": {
            "forget": main_out['forget']["avg_m_entr"],
            "retain": main_out["retain"]["avg_m_entr"],
            "test": main_out["test"]["avg_m_entr"],
            }
            }

    # now need to gather output dists and labels for retain_one and retain_two, for the MIAs
    # Both MIAs recompute entropy and m-entropy, which is a bit of wasted computation,
    # could refactor that later
    # at least this way, they do not have to pass through the data loader for forget and test again
    # print("Running through `retain_one` and `retain_two` for MIAs...\n")
    print("Gathering data for MIAs...\n")
    num_forget_samples = len(main_out["forget"]["probs"][0]) # first dimension gives you number of samples, second dim is the num of classes
    g = torch.Generator().manual_seed(seed)
    retain_probs = main_out["retain"]["probs"]
    test_probs = main_out["test"]["probs"]
    MIA_member_train_probs = retain_probs[torch.randperm(retain_probs.shape[0], generator=g)[:num_forget_samples]]
    chosen_test_probs = test_probs[torch.randperm(test_probs.shape[0], generator=g)[:(2*num_forget_samples)]] # need twice as many so we can cut it in half
    MIA_nonmember_train_probs = chosen_test_probs[:num_forget_samples]
    MIA_nonmember_test_probs = chosen_test_probs[num_forget_samples:]

    # MIA_member_train_out = validate(dataloaders["MIA_member_train"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    # # retain_two_out = validate(dataloaders["retain_two"], model, criterion = criterion, print_freq = 100_000, device = device, w_and_b=False)
    # main_out.update({
    #     "retain_one": retain_one_out,
    #     "retain_two": retain_two_out,
    # })
    
    print("Performing threshold MIA attack...\n")
    attack_results = entropy_threshold_MIA(
        train_member_probs = MIA_member_train_probs, 
        train_non_member_probs = MIA_nonmember_train_probs, 
        forget_probs = main_out['forget']["probs"], 
        audit_non_member_probs = MIA_nonmember_test_probs
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

    return results, main_out



def measure_solo_and_comparison_metrics(model, dataloaders, device, seed, retrain_out_path = None, bad_teacher = None, base_out_path = None, num_classes = None, retrain_model = None, base_model = None, compute_fisher = False, bound_info = None):
    """
    Measure ALL unlearning metrics, including those which require a reference model for comparison

    compute_fisher: if True, also compute the residual-information metrics in
    evaluation/residual_information.py (information_score, efficacy, efficacy_upper_bound, and --
    if `bound_info` is also given -- fisher_information_bound). This requires retrain_out_path to
    point to a `retrained_out.pth` that was itself generated with compute_fisher=True (i.e. it has
    a "fisher_diag" entry under "retain"), since fisher_information_bound needs the retrained
    model's Fisher information over the retain set.

    bound_info: the dict returned by unlearn.fisher.fisher() (only produced when `model` was
    unlearned via Fisher scrubbing). Needed for fisher_information_bound; the other residual
    information metrics don't use it.
    """

    # do the first pass of solo metrics on your main model (usually the unlearned one)
    main_results, main_out = measure_solo_metrics(model, dataloaders, device, seed = seed, compute_fisher = compute_fisher)
    # load your reference out object (usually retrain from scratch)
    retrain_out = torch.load(retrain_out_path)
    # ensure bad teacher is in eval mode
    bad_teacher.eval()
    # load original model out object
    base_out = torch.load(base_out_path)

    # Tug-of-War metric
    da_forget = abs(main_results["acc"]["forget"] - retrain_out["forget"]["avg_acc"]) / 100
    da_retain = abs(main_results["acc"]["retain"] - retrain_out["retain"]["avg_acc"]) / 100
    da_test = abs(main_results["acc"]["test"] - retrain_out["test"]["avg_acc"]) / 100

    ToW = (1 - da_forget) * (1 - da_retain) * (1 - da_test)
    main_results.update({
        "ToW": ToW
    })

    print(f"Evaluating differences/distances between unlearned, retrain, and bad_teacher outputs ...\n")
    
    # get bad teacher outputs for ZRF
    bad_teacher_forget_out = validate(dataloaders["forget"], bad_teacher, criterion = nn.CrossEntropyLoss(), print_freq = 100_000, device = device, w_and_b=False)


    # getting predictions (useful for a couple metrics)
    unlearned_preds = main_out['forget']['probs'].argmax(dim = 1)
    retrain_preds = retrain_out['forget']['probs'].argmax(dim = 1)
    base_preds = base_out['forget']['probs'].argmax(dim = 1)
    true_forget_labels = main_out["forget"]["targets"]

    main_results.update({ 

        "outputs": {
            
            f"retrain_vs_unlearned": {
                "forget": {
                    "absolute_distance": absolute_distance( main_out['forget']['probs'], retrain_out['forget']['probs'] ),
                    "l2_distance": l2_distance( main_out['forget']['probs'], retrain_out['forget']['probs']),
                    "JS_divergence": JSDiv( main_out['forget']['probs'], retrain_out['forget']['probs']),
                    "prediction_distribution_diff": pred_distribution_difference(unlearned_preds, retrain_preds),
                    "normalized_confusion_distance": normalized_confusion_distance(unlearned_preds, retrain_preds, base_preds, true_forget_labels, num_classes = num_classes)
                    },
                "retain": {
                    "absolute_distance": absolute_distance( main_out['retain']['probs'], retrain_out['retain']['probs'] ),
                    },
                "test":{
                    "absolute_distance": absolute_distance( main_out['test']['probs'], retrain_out['test']['probs'] ),
                    },
                "forget_test_avg": {
                    "KL_divergence": kl_div_metric( retrain_out['forget']['probs'], main_out['forget']['probs'], retrain_out['retain']['probs'], main_out['retain']['probs'] )
                    }
                },
            f"bad_teacher_vs_unlearned": {
                "forget": {
                    "ZRF_score": 1 - JSDiv( main_out['forget']['probs'], bad_teacher_forget_out['probs']),
                    },
                },
            f"unlearned": {
                "forget_vs_test": {
                    "wasserstein_distance": wasserstein_distance(main_out['forget']["losses"], main_out['test']["losses"]),
                    "ks_statistics": ks_2samp(main_out['forget']["losses"], main_out['test']["losses"], method = 'asymp')[0] # only need first item, second item is the p-value
                    }
                    }
                },

        "weight_differences": {
            "retrain_vs_unlearned": {
                "l2_distance": weight_distance(model, retrain_model, type = "l2", layer_wise = False)
                },
            "original_vs_unlearned": {
                "l2_distance": weight_distance(model, base_model, type = "l2", layer_wise = False)
                }
            },
            })

    if compute_fisher:
        forget_fisher_diag = main_out["forget"]["fisher_diag"]
        forget_grad_norm = main_out["forget"]["grad_norm"]

        residual_information = {
            "information_score": information_score(forget_fisher_diag),
            "efficacy": efficacy(forget_fisher_diag),
            "efficacy_upper_bound": efficacy_upper_bound(forget_grad_norm),
        }

        retrain_fisher_diag = retrain_out.get("retain", {}).get("fisher_diag")
        if retrain_fisher_diag is not None:
            if bound_info is not None:
                # Fisher-scrubbed model: Sigma_o is the *exact* injected noise variance
                w_o = bound_info["pre_scrub_state_dict"]
                sigma_o = bound_info["noise_variance"]
            else:
                # any other unlearning method: no explicit noise term, so approximate Sigma_o the
                # same way Sigma_r is approximated -- via the retain-set Fisher at the model's own
                # weights, already computed above as part of this same compute_fisher=True pass
                epsilon = 1e-8
                named_params = list(model.named_parameters())
                retain_fisher_diag = main_out["retain"]["fisher_diag"]
                w_o = {name: param.detach() for name, param in named_params}
                sigma_o = {
                    name: 1.0 / (retain_fisher_diag[i] + epsilon)
                    for i, (name, _) in enumerate(named_params)
                }

            residual_information["fisher_information_bound"] = fisher_information_bound(
                w_o, sigma_o, retrain_model, retrain_fisher_diag
            )

        main_results["residual_information"] = residual_information

    return main_results, main_out
    

    


