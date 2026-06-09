import os
import time

import matplotlib.pyplot as plt
import numpy as np
# import pruner
import torch
# from pruner import extract_mask, prune_model_custom, remove_prune
from evaluation.utils import measure_unlearning_metrics
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
# from .RL import RL
from .FT import FT, FT_l1
# from .fisher import fisher, fisher_new
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


def raw():
    pass


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "raw":
        return raw
    # elif name == "RL":
    #     return RL
    elif name == "GA":
        return GA
    elif name == "FT":
        return FT
    elif name == "FT_l1":
        return FT_l1
    # elif name == "fisher":
    #     return fisher
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
        
        # used to be inside `config`
        num_epochs,
        unlearning_lr,
        measure_every,
        device,

        method,
        model,
        dataloaders,
        run,
        forget_set_type,
        unlearning_item,
        w_and_b,
        save_checkpoints_at,
        checkpoint_subfolder,
        print_freq,
        blank_model,
        seed
        ):
    """
    Router for executing unlearning

    `method` is a function, doing the method in question

    This handles all the auxiliary logic (creating results folders, measuring results, saving out, etc)
    """

    method_func = get_unlearn_method(method)
    print(f"Executing unlearning with {method}...\n")

    # if we're doing any of the knowledge distillation methods, then there's some extra set up we need outside the epoch loop
    if method in ["bad_teacher"]:
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

        # `model` here is the student model, which at this time is a copy of the base model, i.e. the full_trained teacher, so we can init the good teacher as such
        full_trained_teacher = copy.deepcopy(model).to(device)
        
        # we need the unlearning teacher to be the same model architecture
        # initialized in the same way as the full_trained_teacher, but not trained
        # (we pass it as an argument for convenience)
        unlearning_teacher = blank_model.to(device)

        # teachers are in eval mode, student is in train
        full_trained_teacher.eval()
        unlearning_teacher.eval()
        opt = torch.optim.Adam(model.parameters(), lr = unlearning_lr)

    else:

        # set up unlearning criterion and opt
        # --- we by default presume SGD, but might want to make this more flexible later
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=unlearning_lr)

        # we do NOT set model.train() otherwise - we would lose Batchnorm statistics that way


    # create save folder if it doesn't exist
    results_folder = init_folder_if_not_exists( f"{base_results_folder}/{method}" )
    
    # init total unlearning time
    total_unlearning_time_ms = 0

    # For each epoch ...
    for k in range(1, num_epochs+1):

        # ... start the clock
        start_time = time.time()

        # ... do one epoch of the unlearning method,
        print(f"---------- Epoch {k}\n")

        # ... --- the args for bad teacher and other knowledge distillation methods are different
        if method == "bad_teacher":
            # we ONLY set to train on knowledge distillation unlearning
            # MAYBE COME BACK TO THIS
            model.train()
            model, _ = method_func(unlearning_loader, model, unlearning_teacher, full_trained_teacher, opt, epoch = k, print_freq = print_freq, device = device, w_and_b = w_and_b)
        else:
            model.eval()
            model, _ = method_func(dataloaders, model, criterion, opt, epoch = k, print_freq = print_freq, device = device, w_and_b = w_and_b)
        

        # ... stop clock for this epoch
        epoch_duration = time.time() - start_time
        total_unlearning_time_ms += epoch_duration * 1000  # Convert to milliseconds


        # ... and if we're measuring results this epoch...,
        if k % measure_every == 0:

            # make sure we're in eval mode, regardless of whether we are during unlearning
            model.eval()
            
            # ... create an epoch-level subfolder if it doesn't exist
            epoch_results_folder = init_folder_if_not_exists( os.path.join(results_folder, f"epoch_{k}") )

            # ... run some evaluations, and add metadata
            # print(f"[do_unlearning] before measure_unlearning_metrics: model.training = {model.training}")
            results = measure_unlearning_metrics(model = model, dataloaders = dataloaders, device = device)
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
            print(f"Saved epoch {k} results.\n")

        # ... and if we're saving checkpoints out,
        if k in save_checkpoints_at:
            unlearn_checkpoint_path = os.path.join(checkpoint_subfolder, f"{method}_epoch_{k}_{forget_set_type}_{unlearning_item}.pth")
            checkpoint = {
                'epoch': k,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict()
            }
            torch.save(checkpoint, unlearn_checkpoint_path, _use_new_zipfile_serialization = False)


    return model

    


