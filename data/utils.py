"""
    setup model and datasets
"""


import copy
import os
import random

# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
import sys
import time

import numpy as np
import torch
# from data.dataloaders import *
# from data.dataloaders import TinyImageNet
# from imagenet import prepare_data
# from models import *
from torchvision import transforms


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_loader_from_dataset(dataset, batch_size, shuffle=True, num_workers=0):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle
    )


def split_forget_retain(marked_loader, batch_size, shuffle = True, num_workers=0):
    """Splits the marked dataset into two dataloaders: one for the forget set and one for the retain set
    
    `marked_loader` for a training set should set shuffle = True, otherwise False

    This by default "unmarks" the froget set by by undoing the negation applied to targets that originally "marked" them
    """
    
    # make forget set
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = -forget_dataset.targets[marked] - 1
    # sync .labels for datasets like SVHN whose __getitem__ reads .labels not .targets
    if hasattr(forget_dataset, 'labels'):
        forget_dataset.labels = forget_dataset.targets
    forget_loader = get_loader_from_dataset(forget_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # make retain set
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    if hasattr(retain_dataset, 'labels'):
        retain_dataset.labels = retain_dataset.targets
    retain_loader = get_loader_from_dataset(retain_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    print("Forget set:", len(forget_dataset), "items")
    print("Retain set:", len(retain_dataset), "items")
    print("\n")
    return forget_loader, retain_loader



def split_random(dataloader, p, seed, batch_size, shuffle = True, num_workers=0):

    base_dataset = dataloader.dataset

    rng = np.random.RandomState(seed)
    set_two_idx = []
    for i in range(max(base_dataset.targets) + 1):
        class_idx = np.where(base_dataset.targets == i)[0]
        set_two_idx.append(
            rng.choice(class_idx, int(p * len(class_idx)), replace=False)
        )
    set_two_idx = np.hstack(set_two_idx)
    set_one_idx = list(set(range(len(base_dataset))) - set(set_two_idx))

    set_one = copy.deepcopy(base_dataset)
    set_two = copy.deepcopy(base_dataset)

    set_one.data = base_dataset.data[set_one_idx]
    set_one.targets = base_dataset.targets[set_one_idx]
    if hasattr(set_one, 'labels'):
        set_one.labels = set_one.targets

    set_two.data = base_dataset.data[set_two_idx]
    set_two.targets = base_dataset.targets[set_two_idx]
    if hasattr(set_two, 'labels'):
        set_two.labels = set_two.targets

    dataloader_one = get_loader_from_dataset(set_one, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    dataloader_two = get_loader_from_dataset(set_two, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    print("Split one:", len(dataloader_one.dataset), "items")
    print("Split two:", len(dataloader_two.dataset), "items\n")

    return dataloader_one, dataloader_two









# __all__ = [
#     "setup_model_dataset",
#     "AverageMeter",
#     "warmup_lr",
#     "save_checkpoint",
#     "setup_seed",
#     "accuracy",
# ]



# def save_checkpoint(
#     state, is_SA_best, save_path, pruning, filename="checkpoint.pth.tar"
# ):
#     filepath = os.path.join(save_path, str(pruning) + filename)
#     torch.save(state, filepath)
#     if is_SA_best:
#         shutil.copyfile(
#             filepath, os.path.join(save_path, str(pruning) + "model_SA_best.pth.tar")
#         )


# def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
#     filepath = os.path.join(save_path, str(pruning) + filename)
#     if os.path.exists(filepath):
#         print("Load checkpoint from:{}".format(filepath))
#         return torch.load(filepath, device)
#     print("Checkpoint not found! path:{}".format(filepath))
#     return None



# def dataset_convert_to_train(dataset):
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     )
#     while hasattr(dataset, "dataset"):
#         dataset = dataset.dataset
#     dataset.transform = train_transform
#     dataset.train = False


# def dataset_convert_to_test(dataset, args=None):
#     if args["dataset"] == "TinyImagenet":
#         test_transform = transforms.Compose([])
#     else:
#         test_transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#             ]
#         )
#     while hasattr(dataset, "dataset"):
#         dataset = dataset.dataset
#     dataset.transform = test_transform
#     dataset.train = False


# def setup_model_dataset(dataset = "cifar10", args):
#     if dataset == "cifar10":
#         # classes = 10
#         # normalization = NormalizeByChannelMeanStd(
#         #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
#         # )
#         train_full_loader, val_loader, _ = cifar10_dataloaders(
#             batch_size=args["batch_size"], data_dir="data/CIFAR10", num_workers=args["workers"]
#         )
#         marked_loader, _, test_loader = cifar10_dataloaders(
#             batch_size=args["batch_size"],
#             data_dir="data/CIFAR10",
#             num_workers=args["workers"],
#             class_to_replace=args["class_to_replace"],
#             num_indexes_to_replace=args["num_indexes_to_replace"],
#             indexes_to_replace=args["indexes_to_replace"],
#             seed=args["seed"],
#             only_mark=True,
#             shuffle=True,
#             no_aug=args["no_aug"],
#         )

#         # if args["train_seed"] is None:
#         #     args["train_seed"] = args["seed"]
#         # setup_seed(args["train_seed"])

#         # if args.imagenet_arch:
#         #     model = model_dict[args.arch](num_classes=classes, imagenet=True)
#         # else:
#         #     model = model_dict[args.arch](num_classes=classes)

#         # setup_seed(args["train_seed"])

#         # model.normalize = normalization
#         # return model, train_full_loader, val_loader, test_loader, marked_loader
#         return train_full_loader, val_loader, test_loader, marked_loader
    # elif args["dataset"] == "svhn":
    #     classes = 10
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
    #     )
    #     train_full_loader, val_loader, _ = svhn_dataloaders(
    #         batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
    #     )
    #     marked_loader, _, test_loader = svhn_dataloaders(
    #         batch_size=args.batch_size,
    #         data_dir=args.data,
    #         num_workers=args.workers,
    #         class_to_replace=args.class_to_replace,
    #         num_indexes_to_replace=args.num_indexes_to_replace,
    #         indexes_to_replace=args.indexes_to_replace,
    #         seed=args.seed,
    #         only_mark=True,
    #         shuffle=True,
    #     )
    #     if args.imagenet_arch:
    #         model = model_dict[args.arch](num_classes=classes, imagenet=True)
    #     else:
    #         model = model_dict[args.arch](num_classes=classes)

    #     model.normalize = normalization
    #     return model, train_full_loader, val_loader, test_loader, marked_loader
    # elif args["dataset"] == "cifar100":
    #     classes = 100
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
    #     )
    #     train_full_loader, val_loader, _ = cifar100_dataloaders(
    #         batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
    #     )
    #     marked_loader, _, test_loader = cifar100_dataloaders(
    #         batch_size=args.batch_size,
    #         data_dir=args.data,
    #         num_workers=args.workers,
    #         class_to_replace=args.class_to_replace,
    #         num_indexes_to_replace=args.num_indexes_to_replace,
    #         indexes_to_replace=args.indexes_to_replace,
    #         seed=args.seed,
    #         only_mark=True,
    #         shuffle=True,
    #         no_aug=args.no_aug,
    #     )
    #     if args.imagenet_arch:
    #         model = model_dict[args.arch](num_classes=classes, imagenet=True)
    #     else:
    #         model = model_dict[args.arch](num_classes=classes)
    #     model.normalize = normalization
    #     return model, train_full_loader, val_loader, test_loader, marked_loader
    # elif args["dataset"] == "TinyImagenet":
    #     classes = 200
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    #     train_full_loader, val_loader, test_loader = TinyImageNet(args).data_loaders(
    #         batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
    #     )
    #     # train_full_loader, val_loader, test_loader =None, None,None
    #     marked_loader, _, _ = TinyImageNet(args).data_loaders(
    #         batch_size=args.batch_size,
    #         data_dir=args.data,
    #         num_workers=args.workers,
    #         class_to_replace=args.class_to_replace,
    #         num_indexes_to_replace=args.num_indexes_to_replace,
    #         indexes_to_replace=args.indexes_to_replace,
    #         seed=args.seed,
    #         only_mark=True,
    #         shuffle=True,
    #     )
    #     if args.imagenet_arch:
    #         model = model_dict[args.arch](num_classes=classes, imagenet=True)
    #     else:
    #         model = model_dict[args.arch](num_classes=classes)

    #     model.normalize = normalization
    #     return model, train_full_loader, val_loader, test_loader, marked_loader

    # elif args["dataset"] == "imagenet":
    #     classes = 1000
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #     )
    #     train_ys = torch.load(args.train_y_file)
    #     val_ys = torch.load(args.val_y_file)
    #     model = model_dict[args.arch](num_classes=classes, imagenet=True)

    #     model.normalize = normalization
    #     if args.class_to_replace is None:
    #         loaders = prepare_data(dataset="imagenet", batch_size=args.batch_size)
    #         train_loader, val_loader = loaders["train"], loaders["val"]
    #         return model, train_loader, val_loader
    #     else:
    #         train_subset_indices = torch.ones_like(train_ys)
    #         val_subset_indices = torch.ones_like(val_ys)
    #         train_subset_indices[train_ys == args.class_to_replace] = 0
    #         val_subset_indices[val_ys == args.class_to_replace] = 0
    #         loaders = prepare_data(
    #             dataset="imagenet",
    #             batch_size=args.batch_size,
    #             train_subset_indices=train_subset_indices,
    #             val_subset_indices=val_subset_indices,
    #         )
    #         retain_loader = loaders["train"]
    #         forget_loader = loaders["fog"]
    #         val_loader = loaders["val"]
    #         return model, retain_loader, forget_loader, val_loader

    # elif args["dataset"] == "cifar100_no_val":
    #     classes = 100
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]
    #     )
    #     train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
    #         batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
    #     )

    # elif args["dataset"] == "cifar10_no_val":
    #     classes = 10
    #     normalization = NormalizeByChannelMeanStd(
    #         mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
    #     )
    #     train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
    #         batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers
    #     )

    # else:
    #     raise ValueError("Dataset not supported yet!")
    # # import pdb;pdb.set_trace()

    # # if args.imagenet_arch:
    # #     model = model_dict[args.arch](num_classes=classes, imagenet=True)
    # # else:
    # #     model = model_dict[args.arch](num_classes=classes)

    # # model.normalize = normalization
    # # return model, train_set_loader, val_loader, test_loader
    # return train_set_loader, val_loader, test_loader




# class NormalizeByChannelMeanStd(torch.nn.Module):
#     def __init__(self, mean, std):
#         super(NormalizeByChannelMeanStd, self).__init__()
#         if not isinstance(mean, torch.Tensor):
#             mean = torch.tensor(mean)
#         if not isinstance(std, torch.Tensor):
#             std = torch.tensor(std)
#         self.register_buffer("mean", mean)
#         self.register_buffer("std", std)

#     def forward(self, tensor):
#         return self.normalize_fn(tensor, self.mean, self.std)

#     def extra_repr(self):
#         return "mean={}, std={}".format(self.mean, self.std)

#     def normalize_fn(self, tensor, mean, std):
#         """Differentiable version of torchvision.functional.normalize"""
#         # here we assume the color channel is in at dim=1
#         mean = mean[None, :, None, None]
#         std = std[None, :, None, None]
#         return tensor.sub(mean).div(std)




# def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
#     if len(commands) == 0:
#         return
#     if os.path.exists(dir):
#         shutil.rmtree(dir)
#     if shuffle:
#         random.shuffle(commands)
#         random.shuffle(gpus)
#     os.makedirs(dir, exist_ok=True)

#     fout = open("stop_{}.sh".format(dir), "w")
#     print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
#     fout.close()

#     n_gpu = len(gpus)
#     for i, gpu in enumerate(gpus):
#         i_commands = commands[i::n_gpu]
#         if len(i_commands) == 0:
#             continue
#         prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

#         sh_path = os.path.join(dir, "run{}.sh".format(i))
#         fout = open(sh_path, "w")
#         for com in i_commands:
#             print(prefix + com, file=fout)
#         fout.close()
#         if call:
#             os.system("bash {}&".format(sh_path))
#             time.sleep(delay)



# def get_poisoned_loader(poison_loader, unpoison_loader, test_loader, poison_func, args):
#     poison_dataset = copy.deepcopy(poison_loader.dataset)
#     poison_test_dataset = copy.deepcopy(test_loader.dataset)

#     poison_dataset.data, poison_dataset.targets = poison_func(
#         poison_dataset.data, poison_dataset.targets
#     )
#     poison_test_dataset.data, poison_test_dataset.targets = poison_func(
#         poison_test_dataset.data, poison_test_dataset.targets
#     )

#     full_dataset = torch.utils.data.ConcatDataset(
#         [unpoison_loader.dataset, poison_dataset]
#     )

#     poisoned_loader = get_loader_from_dataset(
#         poison_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
#     )
#     poisoned_full_loader = get_loader_from_dataset(
#         full_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
#     )
#     poisoned_test_loader = get_loader_from_dataset(
#         poison_test_dataset, batch_size=args.batch_size, seed=args.seed, shuffle=False
#     )

#     return poisoned_loader, unpoison_loader, poisoned_full_loader, poisoned_test_loader



# def retain_to_train_and_val(
#     retain_loader,
#     seed: int = 1,
# ):


#     retain_dataset = retain_loader.dataset
#     batch_size = retain_loader.batch_size
    
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1),
#             transforms.RandomRotation(degrees = 10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2023, 0.1994, 0.2010]), 
#         ]
#         )
    
#     test_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]), 
#         ]
#         )
#     # -------------------------------------------------------------------- #
#     # ----- repeating as much as possible from `cifar10_dataloaders` ----- #
#     # -------------------------------------------------------------------- #

#     retain_dataset.targets = np.array(retain_dataset.targets)
    
#     rng = np.random.RandomState(seed)
#     valid_set = copy.deepcopy(retain_dataset)
#     valid_idx = []
#     for i in range(max(retain_dataset.targets) + 1):
#         class_idx = np.where(retain_dataset.targets == i)[0]
#         valid_idx.append(
#             rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
#         )
#     valid_idx = np.hstack(valid_idx)
#     train_set = copy.deepcopy(retain_dataset)

#     valid_set.data = train_set.data[valid_idx]
#     valid_set.targets = train_set.targets[valid_idx]

#     train_idx = list(set(range(len(retain_dataset))) - set(valid_idx))
#     train_set.data = train_set.data[train_idx]
#     train_set.targets = train_set.targets[train_idx]

#     # assign transforms
#     train_set.transform = train_transform
#     valid_set.transform = test_transform

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=0
#         # worker_init_fn=_init_fn if seed is not None else None,
#     )
#     val_loader = DataLoader(
#         valid_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0
#         # worker_init_fn=_init_fn if seed is not None else None,
#     )
#     print("SPLITTING RETAIN SET INTO TRAIN AND VAL:\n")
#     print(f"Dataset information: CIFAR-10\t {len(train_set)} images for training \t {len(valid_set)} images for validation\t")

#     return train_loader, val_loader
