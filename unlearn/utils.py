import os
import time

import matplotlib.pyplot as plt
import numpy as np
# import pruner
import torch
# from pruner import extract_mask, prune_model_custom, remove_prune
from evaluation.utils import measure_solo_and_comparison_metrics, measure_solo_metrics
import wandb
from trainer.utils import init_folder_if_not_exists
from torch.utils.data import DataLoader
from data.utils import split_random
import copy


# def plot_training_curve(training_result, save_dir, prefix):
#     # plot training curve
#     for name, result in training_result.items():
#         plt.plot(result, label=f"{name}_acc")
#     plt.legend()
#     plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
#     plt.close()


# def save_unlearn_checkpoint(model, evaluation_result, args):
#     state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
#     utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
#     utils.save_checkpoint(
#         evaluation_result,
#         False,
#         args.save_dir,
#         args.unlearn,
#         filename="eval_result.pth.tar",
    # )


# def load_unlearn_checkpoint(model, device, args):
#     checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
#     if checkpoint is None or checkpoint.get("state_dict") is None:
#         return None

#     current_mask = pruner.extract_mask(checkpoint["state_dict"])
#     pruner.prune_model_custom(model, current_mask)
#     pruner.check_sparsity(model)

#     model.load_state_dict(checkpoint["state_dict"])

#     # adding an extra forward process to enable the masks
#     x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
#     model.eval()
#     with torch.no_grad():
#         model(x_rand)

#     evaluation_result = checkpoint.get("evaluation_result")
#     return model, evaluation_result


# def _iterative_unlearn_impl(unlearn_iter_func):
#     def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
#         total_unlearn_time_ms = 0 #Initialize total unlearn time

#         decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
#         if args.rewind_epoch != 0:
#             initialization = torch.load(
#                 args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
#             )
#             current_mask = extract_mask(model.state_dict())
#             remove_prune(model)
#             # weight rewinding
#             # rewind, initialization is a full model architecture without masks
#             model.load_state_dict(initialization, strict=True)
#             prune_model_custom(model, current_mask)
    
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             args.unlearn_lr,
#             momentum=args.momentum,
#             weight_decay=args.weight_decay,
#         )

#         if args.imagenet_arch and args.unlearn == "retrain":
#             lambda0 = (
#                 lambda cur_iter: (cur_iter + 1) / args.warmup
#                 if cur_iter < args.warmup
#                 else (
#                     0.5
#                     * (
#                         1.0
#                         + np.cos(
#                             np.pi
#                             * (
#                                 (cur_iter - args.warmup)
#                                 / (args.unlearn_epochs - args.warmup)
#                             )
#                         )
#                     )
#                 )
#             )
#             scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
#         else:
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                 optimizer, milestones=decreasing_lr, gamma=0.1
#             )  # 0.1 is fixed
#         if args.rewind_epoch != 0:
#             # learning rate rewinding
#             for _ in range(args.rewind_epoch):
#                 scheduler.step()
#         callback = kwargs.get('callback', None)

#         for epoch in range(0, args.unlearn_epochs):
#             start_time = time.time()

#             print(
#                 "Epoch #{}, Learning rate: {}".format(
#                     epoch, optimizer.state_dict()["param_groups"][0]["lr"]
#                 )
#             )

#             train_acc = unlearn_iter_func(
#                 data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
#             )
#             scheduler.step()
#             epoch_duration = time.time() - start_time
#             total_unlearn_time_ms += epoch_duration * 1000  # Convert to milliseconds

#             print("one epoch duration:{}".format(time.time() - start_time))
            
#             # Evaluate every 2 epochs
#             if callback is not None and ((epoch + 1) % 2 == 0 or (epoch + 1) == args.unlearn_epochs):
#                 callback(data_loaders, epoch + 1, model, args, criterion, total_unlearn_time_ms, **kwargs)

#     return _wrapped


# def iterative_unlearn(func):
#     """usage:

#     @iterative_unlearn

#     def func(data_loaders, model, criterion, optimizer, epoch, args)"""
#     return _iterative_unlearn_impl(func)









# def GA_regimen(
#         model, 
#         dataloaders, 
#         num_epochs, 
#         criterion, 
#         opt, 
#         print_freq, 
#         measure_every, 
#         device, 
#         run,
#         forget_set_type,
#         unlearning_item, 
#         results_folder,
#         w_and_b,
#         save_checkpoints_at, 
#         checkpoint_subfolder):

    

#     # For each epoch ...
#     for k in range(1, num_epochs+1):

#         # ... do one round of GA,
#         model, top1_avg = GA(dataloaders["forget"], model, criterion, opt, epoch = k, print_freq = print_freq, device = device, w_and_b = w_and_b)

#         # ... run some evaluations
#         results = measure_unlearning_metrics(model = model, dataloaders = dataloaders, device = device)
#         results.update({
#             "type": "unlearn",
#             "epoch": k,
#             "run": run,
#             "forget_set_type": forget_set_type,
#             "unlearning_item": unlearning_item,
#             "method": "GA",
#         })

#         # ... and if we're saving results out,
#         if k % measure_every == 0:
            
#             # ... create an epoch-level subfolder if it doesn't exist
#             epoch_results_folder = os.path.join(results_folder, f"epoch_{k}")
#             if not os.path.exists(epoch_results_folder):   
#                 print(f"{epoch_results_folder} doesn't exist - creating it...\n")
#                 os.makedirs(epoch_results_folder, exist_ok=True)

#             # ... and save them
#             if w_and_b:
#                 wandb.log(results)
#             with open(os.path.join(epoch_results_folder, f"{forget_set_type}_{unlearning_item}.json"), "w") as f:
#                 json.dump(results, f, indent=4)
#             print(f"Saved epoch {k} results.\n")

#         # ... and if we're saving checkpoints out,
#         if k in save_checkpoints_at:
#             unlearn_checkpoint_path = os.path.join(checkpoint_subfolder, f"GA_epoch_{k}_{forget_set_type}_{unlearning_item}.pth")
#             checkpoint = {
#                 'epoch': k,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': opt.state_dict()
#             }
#             torch.save(checkpoint, unlearn_checkpoint_path, _use_new_zipfile_serialization = False)


#     return model



from .GA import GA
# from .GA import GA_l1
from .RL import RL, setup_RL_loader
from .FT import FT, FT_l1
from .NegGrad_plus import NegGrad_plus
from .fisher import fisher
# from .retrain import retrain
# from .impl import load_unlearn_checkpoint, save_unlearn_checkpoint
# from .Wfisher import Wfisher
# from .FT_prune import FT_prune
# from .FT_prune_bi import FT_prune_bi
# from .GA_prune_bi import GA_prune_bi
# from .GA_prune import GA_prune

# from .RL_pro import RL_proximal
# from .boundary_ex import boundary_expanding
from .boundary_sh import boundary_shrink
from .bad_teacher import bad_teacher, BadTeacherUnLearningData
from .scrub import scrub, DistillKL
from .UNSIR import UNSIR, UNSIR_noise, UNSIR_noise_train, UNSIR_create_noisy_loader

def raw():
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    elif name == "RL":
        return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    elif name == "NegGrad_plus":
        return NegGrad_plus
    elif name == "fisher":
        return fisher
    # elif name == "retrain":
    #     return retrain
    # elif name == "fisher_new":
    #     return fisher_new
    # elif name == "wfisher":
    #     return Wfisher
    # elif name == "FT_prune":
    #     return FT_prune
    # elif name == "FT_prune_bi":
    #     return FT_prune_bi
    # elif name == "GA_prune":
    #     return GA_prune
    # elif name == "GA_prune_bi":
    #     return GA_prune_bi
    # elif name == "GA_l1":
    #     return GA_l1
    # elif name == "boundary_expanding":
    #     return boundary_expanding
    elif name == "boundary_shrink":
        return boundary_shrink
    elif name == "bad_teacher":
        return bad_teacher
    elif name == "scrub":
        return scrub
    elif name == "UNSIR":
        return UNSIR
    # elif name == "RL_proximal":
    #     return RL_proximal
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")

import json
import numpy as np
import torch.nn as nn
import torch
import time
def do_unlearning(
        base_results_folder,
        method_hyperparams,
        device,

        method,
        model,
        dataloaders,
        run,
        forget_set_type,
        unlearning_item,
        w_and_b,
        checkpoint_subfolder,
        blank_model,
        seed,
        retrain_out_path = None
        ):
    """
    Router for executing unlearning

    `method` is a function, doing the method in question

    This handles all the auxiliary logic (creating results folders, measuring results, saving out, etc)
    """

    method_func = get_unlearn_method(method)
    print(f"Executing unlearning with {method}...\n")

    # establish optimizer first, since we need it in all forks of the if-else later
    if method_hyperparams["opt"] == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=method_hyperparams["lr"], weight_decay=method_hyperparams.get("weight_decay", 0))
    elif method_hyperparams["opt"] == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=method_hyperparams["lr"], weight_decay=method_hyperparams.get("weight_decay", 0))
    else:
        # default to SGD 
        opt = torch.optim.SGD(model.parameters(), lr=method_hyperparams["lr"], weight_decay=method_hyperparams.get("weight_decay", 0))

        

    # if we're doing any of the knowledge distillation methods, then there's some extra set up we need outside the epoch loop
    if method == "bad_teacher":
        # need to make unlearning dataset
        retain_loader = dataloaders["retain"]
        retain_keep_loader, _ = split_random(dataloaders["retain"], p = .3, seed = seed, batch_size=retain_loader.batch_size, shuffle = False)
        unlearning_data = BadTeacherUnLearningData(forget_data = dataloaders["forget"].dataset, retain_data = retain_keep_loader.dataset)
        unlearning_loader = DataLoader(
            unlearning_data, 
            batch_size = retain_loader.batch_size, 
            shuffle=True, 
            num_workers=retain_loader.num_workers, 
            pin_memory=True
            )

        full_trained_teacher = copy.deepcopy(model).to(device) # `model` here is the student model, which at this time is a copy of the base model, i.e. the full_trained teacher, so we can init the good teacher as such
        unlearning_teacher = blank_model.to(device) # we need the unlearning teacher to be the same model architecture, initialized in the same way as the full_trained_teacher, but not trained (we pass it as an argument for convenience)
        full_trained_teacher.eval() # teachers are in eval mode, student is in train
        unlearning_teacher.eval()

    if method == "UNSIR":
        
        # Infer noise tensor shape from a sample in the forget loader
        sample_batch = next(iter(dataloaders["forget"]))
        sample_img = sample_batch[0][0]
        noise_batch_size = method_hyperparams.get("noise_batch_size", 128)
        C, H, W = sample_img.shape

        # Train the noise module to maximally confuse the model on the forget class
        noise = UNSIR_noise(noise_batch_size, C, H, W).to(device)
        print("Training UNSIR noise...\n")
        noise = UNSIR_noise_train(
            noise, model,
            forget_class_label=unlearning_item,
            num_epochs=method_hyperparams.get("noise_train_epochs", 5),
            noise_batch_size=noise_batch_size,
            device=device
        )

        retain_dataset = dataloaders["retain"].dataset
        targets = np.array(retain_dataset.targets)
        rng = np.random.RandomState(seed)
        retain_indices = []
        for cls in np.unique(targets):
            cls_indices = np.where(targets == cls)[0]
            chosen = rng.choice(cls_indices, size=min(1000, len(cls_indices)), replace=False)
            retain_indices.extend(chosen)
        retain_samples = [retain_dataset[i] for i in retain_indices]

        # Create the mixed noisy+retain loader used during the impair step
        noisy_loader = UNSIR_create_noisy_loader(
            noise=noise,
            forget_class_label=unlearning_item,
            retain_samples=retain_samples,
            batch_size=dataloaders["retain"].batch_size,
            num_noise_batches=method_hyperparams.get("num_noise_batches", 20),
            num_workers = dataloaders["retain"].num_workers,
            pin_memory = dataloaders["retain"].pin_memory
        )

        dataloaders["UNSIR_noisy_loader"] = noisy_loader

        criterion = nn.CrossEntropyLoss()

    if method == "scrub":

            # set up scrub criterion and opt
            # following their advice in the paper, we default to using Adam, since they say it's best for large-scale data (which is all of ours)
            model_t = copy.deepcopy(model)
            # model_s = copy.deepcopy(model)
            model.to(device) # this IS the "student model"
            model_t.to(device)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1], gamma=0.1)
            criterion_cls = nn.CrossEntropyLoss()
            criterion_div = DistillKL(T = 4) # This T is hardcoded from the original paper, and also used in Di et al 2024

            # rebuild forget/retain loaders with scrub-specific batch sizes
            for key, bs_key in [("forget", "forget_batch_size"), ("retain", "retain_batch_size")]:
                old = dataloaders[key]
                dataloaders[key] = DataLoader(
                    old.dataset,
                    batch_size=method_hyperparams[bs_key],
                    shuffle=isinstance(old.sampler, torch.utils.data.RandomSampler),
                    num_workers=old.num_workers,
                    pin_memory=old.pin_memory,
                )

    else:

        # all others use normal cross-entropy loss
        criterion = nn.CrossEntropyLoss()

        # we do NOT set model.train() otherwise - we would lose Batchnorm statistics that way


    # If method is one of the RL ones, we need some extra data
    if "RL" in method:
        dataloaders["RL_train_loader"] = setup_RL_loader(dataloaders)
        

    # create save folder if it doesn't exist
    results_folder = init_folder_if_not_exists( f"{base_results_folder}/{method}" )
    
    # init total unlearning time
    total_unlearning_time_ms = 0

    # For each epoch ...
    for k in range(1, method_hyperparams["num_epochs"]+1):

        # ... start the clock
        start_time = time.time()

        # ... do one round of the unlearning method
        print(f"---------- Epoch {k}\n")

        # selectively set model to train or eval depending on method
        if method == "bad_teacher":
            model.train()
            model, _ = method_func(unlearning_loader, model, unlearning_teacher, full_trained_teacher, opt, epoch=k, device=device, w_and_b=w_and_b, **method_hyperparams)
        
        elif method == "scrub":
            model.eval()
            model, _ = method_func(dataloaders, model, model_t, criterion_cls, criterion_div, opt, scheduler, epoch=k, device=device, w_and_b=w_and_b, **method_hyperparams)
        elif method == "UNSIR":
            model.train()
            model, _ = method_func(dataloaders, model, criterion, opt, epoch=k, device=device, w_and_b=w_and_b, **method_hyperparams)
        else:
            model.eval()
            model, _ = method_func(dataloaders, model, criterion, opt, epoch=k, device=device, w_and_b=w_and_b, **method_hyperparams)
        

        # ... stop clock for this epoch
        epoch_duration = time.time() - start_time
        total_unlearning_time_ms += epoch_duration * 1000  # Convert to milliseconds


        # ... and if we're measuring results this epoch...,
        if k in method_hyperparams['save_checkpoints_at']:

            # make sure we're in eval mode, regardless of whether we are during unlearning
            model.eval()
            
            # ... create an epoch-level subfolder if it doesn't exist
            epoch_results_folder = init_folder_if_not_exists( os.path.join(results_folder, f"epoch_{k}") )

            # ... run some evaluations, and add metadata
            # print(f"[do_unlearning] before measure_unlearning_metrics: model.training = {model.training}")
            results, unlearned_out = measure_solo_and_comparison_metrics(
                model = model, 
                dataloaders = dataloaders, 
                device = device,
                reference_out_path = retrain_out_path
                )
            # print(f"[do_unlearning] after measure_unlearning_metrics: model.training = {model.training}")
            results.update({
                "type": "unlearn",
                "epoch": k,
                "run": run,
                "forget_set_type": forget_set_type,
                "unlearning_item": unlearning_item,
                "method": method,
                "epoch_duration": epoch_duration,
                "total_unlearning_time_up_to_now": total_unlearning_time_ms
            })

            # ... and save them
            if w_and_b:
                wandb.log(results)
            with open(os.path.join(epoch_results_folder, f"{forget_set_type}_{unlearning_item}.json"), "w") as f:
                json.dump(results, f, indent=4)
            torch.save(unlearned_out, os.path.join(epoch_results_folder, f"{forget_set_type}_{unlearning_item}_out.pth"))
            print(f"Saved epoch {k} results.\n")

            # ... and if we're saving checkpoints out,
        
            unlearn_checkpoint_path = os.path.join(checkpoint_subfolder, f"{method}_epoch_{k}_{forget_set_type}_{unlearning_item}.pth")
            checkpoint = {
                'epoch': k,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }
            torch.save(checkpoint, unlearn_checkpoint_path)


    return model

    


