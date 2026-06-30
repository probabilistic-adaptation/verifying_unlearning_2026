"""
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
"""
import copy
import glob
import os
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder
from tqdm import tqdm
from data.utils import get_loader_from_dataset



# def cifar10_dataloaders(
#     batch_size=128,
#     data_dir="datasets/cifar10",
#     num_workers=0,
#     class_to_replace: int = None,
#     percent_to_replace=None,
#     seed: int = 1,
#     only_mark: bool = False,
#     val = True
# ):
    
#     # --- set transforms for both train and test splits --- #
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1),
#             transforms.RandomRotation(degrees = 10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2023, 0.1994, 0.2010]), 
#         ]
#     )

#     test_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]), 
#         ]
#     )


    
#     # --- load the dataset --- #
#     train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
#     train_set.targets = np.array(train_set.targets)
#     test_set.targets = np.array(test_set.targets)

#     # IF WE NEED A VALIDATION SET, SPLIT IT OFF FROM THE TRAINING SET (10% of each class)
#     # and then reconfigure train set to accommodate
#     if val:
#         rng = np.random.RandomState(seed)
#         valid_set = copy.deepcopy(train_set)
#         valid_idx = []
#         for i in range(max(train_set.targets) + 1):
#             class_idx = np.where(train_set.targets == i)[0]
#             valid_idx.append(
#                 rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
#             )
#         valid_idx = np.hstack(valid_idx)
#         train_set_copy = copy.deepcopy(train_set)

#         valid_set.data = train_set_copy.data[valid_idx]
#         valid_set.targets = train_set_copy.targets[valid_idx]

#         train_idx = list(set(range(len(train_set))) - set(valid_idx))

#         train_set.data = train_set_copy.data[train_idx]
#         train_set.targets = train_set_copy.targets[train_idx]

#     # --- replace some data if specified --- #

#     # ------ cant specify both class and number to replace
#     if class_to_replace is not None and percent_to_replace is not None:
#         raise ValueError("Only one of `class_to_replace` and `percent_to_replace` can be specified")


#     if class_to_replace is not None:
        
#         # replace classes in training, val, and test (modified in place)
#         # --- by default replacing ALL indexes of class
#         replace_class(train_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
#         if val:
#             replace_class(valid_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
#         replace_class(test_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)

#         # if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
#         #     test_set.data = test_set.data[test_set.targets != class_to_replace]
#         #     test_set.targets = test_set.targets[test_set.targets != class_to_replace]

#     if percent_to_replace is not None:
#         # replace some percentage across all classes (modified in place)
#         replace_random_uniform(train_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
#         if val:
#             replace_random_uniform(valid_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
#         replace_random_uniform(test_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)

#     loader_args = {"num_workers": 0, "pin_memory": False}

#     def _init_fn(worker_id):
#         np.random.seed(int(seed))

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         worker_init_fn=_init_fn if seed is not None else None,
#         **loader_args,
#     )
    
#     if val:
#         val_loader = DataLoader(
#             valid_set,
#             batch_size=batch_size,
#             shuffle=False,
#             worker_init_fn=_init_fn if seed is not None else None,
#             **loader_args,
#         )
#     else:
#         val_loader = None

#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         worker_init_fn=_init_fn if seed is not None else None,
#         **loader_args,
#     )

#     # --- print what we did --- #

#     print("="*10 + " DATALOADER INFO")
#     print("Dataset: CIFAR-10")
#     print(f"Train: {len(train_set)} images for training")
#     if val:
#         print(f"Val: {len(valid_set)} images for validation")
#     print(f"Test: {len(test_set)} images for testing")
#     if class_to_replace is not None:
#         print(f"Replaced class {class_to_replace} in train")
#     if percent_to_replace is not None:
#         print(f"Replaced {percent_to_replace:.1f}% of labels across all classes in train")
#     print("Training augmentation = randomcrop(32,4) + randomhorizontalflip + colorjitter + randomrotation + normalize")
#     print("Validation/Test augmentation = normalize")
#     print("\n")

#     return train_loader, val_loader, test_loader


def replace_indexes(
    dataset: torch.utils.data.Dataset, indexes, seed=0, only_mark: bool = False
):
    
    """
    For the given dataset, replace the labels for data at "indexes" with something else:
        if `only_mark`: replace the labels with negative assignments (e.g. 0 -> -1, 1 -> -2, etc)
        if not `only_mark`: replace the labels with the literal label of randomly sampled other data points
    """

    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes)
        )
        dataset.data[indexes] = dataset.data[new_indexes]
        try:
            dataset.targets[indexes] = dataset.targets[new_indexes]
        except:
            try:
                dataset.labels[indexes] = dataset.labels[new_indexes]
            except:
                dataset._labels[indexes] = dataset._labels[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        try:
            dataset.targets[indexes] = -dataset.targets[indexes] - 1
        except:
            try:
                dataset.labels[indexes] = -dataset.labels[indexes] - 1
            except:
                dataset._labels[indexes] = -dataset._labels[indexes] - 1

    print("Replacing indeces:", indexes[:10], "...")



# All `replace` functions take arguments as:
# - dataset
# - value
# - seed
# - only_mark

# in the future, if we want to replace just a subset of a particular class, we need to edit in order to pass both class value and percentage-within to replace


def replace_class(
    dataset: torch.utils.data.Dataset,
    class_to_replace: int,
    # num_indexes_to_replace: int = None,
    seed: int = 0,
    only_mark: bool = False,
):

    """
    Using `replace_indexes` above to replace the labels for an entire class, 
    or a subset of observations from a class.

    `num_indexes_to_replace` needs to be <= the number of labels in the given `classes_to_replace`,
    or, if shuffling the entire dataset, <= number of items in dataset


    """
    # Gather the indeces for the classes you want to replace, either...
    # ... shuffling all classes, 
    if class_to_replace == -1:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.targets))
        except:
            try:
                indexes = np.flatnonzero(np.ones_like(dataset.labels))
            except:
                indexes = np.flatnonzero(np.ones_like(dataset._labels))
    
    # ... or shuffling a specific class.
    else:
        try:
            indexes = np.flatnonzero(np.isin(np.array(dataset.targets), class_to_replace))
        except:
            try:
                indexes = np.flatnonzero(np.isin(np.array(dataset.labels), class_to_replace))
            except:
                indexes = np.flatnonzero(np.isin(np.array(dataset._labels), class_to_replace))
    
    # If you only want to shuffle a subset, then shuffle a random subset
    # if num_indexes_to_replace is not None:
    #     assert num_indexes_to_replace <= len(
    #         indexes
    #     ), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
    #     rng = np.random.RandomState(seed)
    #     indexes = rng.choice(indexes, size=num_indexes_to_replace, replace=False)
    #     print(f"Replacing indexes {indexes}")


    # Now that you've gathered all the relevant indexes for the classes you want to forget, 
    # Literally replaces the indexes you've gathered
    replace_indexes(dataset, indexes, seed, only_mark)


def replace_random_uniform(
    dataset: torch.utils.data.Dataset,
    percent_to_replace: float,
    seed: int = 0,
    only_mark: bool = False,
):
    """
    Replaces a random percentage of labels equally across every class in the dataset.
    
    Args:
        percent_to_replace: Float between 0 and 100 (e.g., 10 for 10% of every class).
    """
    
    # Access labels/targets
    try:
        labels = np.array(dataset.targets)
    except:
        try:
            labels = np.array(dataset.labels)
        except:
            labels = np.array(dataset._labels)

    unique_classes = np.unique(labels)
    all_selected_indexes = []
    rng = np.random.RandomState(seed)

    # Iterate through each class, gather indexes
    for cls in unique_classes:
        
        # print(f"Replacing in class {cls}...")
        class_indexes = np.flatnonzero(labels == cls)
        num_to_replace = int(len(class_indexes) * percent_to_replace)
        # print(f"Replacing {num_to_replace} samples...\n")
        if num_to_replace > 0:
            selected = rng.choice(class_indexes, size=num_to_replace, replace=False)
            all_selected_indexes.extend(selected)
        else:
            print(f"Warning: class {cls} has only {len(class_indexes)} samples, skipping replacement for this class.")

    # Replace accumulated list of indexes
    if all_selected_indexes:
        all_selected_indexes = np.array(all_selected_indexes)
        print(f"Replacing {len(all_selected_indexes)} samples total "
              f"({percent_to_replace*100:.1f}% across all {len(unique_classes)} classes)")
        
        replace_indexes(dataset, all_selected_indexes, seed, only_mark)
    else:
        print("Warning: Percentage too low or dataset too small to replace any samples.")


def replace_random(
    dataset: torch.utils.data.Dataset,
    percent_to_replace: float,
    seed: int = 0,
    only_mark: bool = False,
):
    """
    Replaces a random percentage of labels - not uniform across classes
    
    Args:
        percent_to_replace: Float between 0 and 100 (e.g., 10 for 10% of every class).
    """
    
    # Access labels/targets - DONT NEED THEM
    # try:
    #     labels = np.array(dataset.targets)
    # except:
    #     try:
    #         labels = np.array(dataset.labels)
    #     except:
    #         labels = np.array(dataset._labels)


    try:
        indexes = np.flatnonzero(np.ones_like(dataset.targets))
    except:
        try:
            indexes = np.flatnonzero(np.ones_like(dataset.labels))
        except:
            indexes = np.flatnonzero(np.ones_like(dataset._labels))

    rng = np.random.RandomState(seed)
    all_selected_indexes = []
    
    
    num_to_replace = int(len(indexes) * percent_to_replace)
    if num_to_replace > 0:
        selected = rng.choice(indexes, size=num_to_replace, replace=False)
        all_selected_indexes.extend(selected)

    # Replace accumulated list of indexes
    if all_selected_indexes:
        all_selected_indexes = np.array(all_selected_indexes)
        print(f"Replacing {len(all_selected_indexes)} samples total ({percent_to_replace*100:.1f}%)")
        replace_indexes(dataset, all_selected_indexes, seed, only_mark)
    else:
        print("Warning: Percentage too low or dataset too small to replace any samples.")




def get_replace_type(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "class":
        return replace_class
    elif name == "random":
        return replace_random
    elif name == "random_uniform":
        return replace_random_uniform
    else:
        raise NotImplementedError(f"Replace type {name} not implemented!")

def unmark_dataset(dataset):
    """
    Finds any negative labels in the dataset and reverts them 
    back to their original positive class index.
    """
    # Locate the correct attribute for labels
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, '_labels'):
        labels = dataset._labels
    else:
        raise AttributeError("Could not find labels attribute in dataset.")

    # Apply the inverse transformation only to marked (negative) labels
    marked_mask = labels < 0
    labels[marked_mask] = -labels[marked_mask] - 1



def get_targets(dataset):
        for attr in ("targets", "labels", "_labels"):
            if hasattr(dataset, attr):
                return np.array(getattr(dataset, attr))
        raise AttributeError(f"No targets attribute found on {type(dataset).__name__}")

# def generic_dataloaders(
#     train_set, 
#     test_set, 
#     batch_size=128,
#     num_workers=0,
#     class_to_replace: int = None,
#     percent_to_replace=None,
#     seed: int = 1,
#     only_mark: bool = False,
#     val = True):


#     # access labels/targets, and just call them "targets", ...
#     train_set.targets = get_targets(train_set)
#     test_set.targets = get_targets(test_set)
#     # sync .labels for datasets like SVHN whose __getitem__ reads .labels not .targets
#     if hasattr(train_set, 'labels'):
#         train_set.labels = train_set.targets
#     if hasattr(test_set, 'labels'):
#         test_set.labels = test_set.targets

#     # IF WE NEED A VALIDATION SET, SPLIT IT OFF FROM THE TRAINING SET (10% of each class)
#     # and then reconfigure train set to accommodate
#     if val:
#         rng = np.random.RandomState(seed)
#         valid_set = copy.deepcopy(train_set)
#         valid_idx = []
#         for i in range(max(train_set.targets) + 1):
#             class_idx = np.where(train_set.targets == i)[0]
#             valid_idx.append(
#                 rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
#             )
#         valid_idx = np.hstack(valid_idx)
#         train_set_copy = copy.deepcopy(train_set)

#         valid_set.data = train_set_copy.data[valid_idx]
#         valid_set.targets = train_set_copy.targets[valid_idx]
#         if hasattr(valid_set, 'labels'):
#             valid_set.labels = valid_set.targets

#         train_idx = list(set(range(len(train_set))) - set(valid_idx))

#         train_set.data = train_set_copy.data[train_idx]
#         train_set.targets = train_set_copy.targets[train_idx]
#         if hasattr(train_set, 'labels'):
#             train_set.labels = train_set.targets

#         # --- replace some data if specified --- #

#     # ------ cant specify both class and number to replace
#     if class_to_replace is not None and percent_to_replace is not None:
#         raise ValueError("Only one of `class_to_replace` and `percent_to_replace` can be specified")


#     if class_to_replace is not None:

#         # replace classes in training, val, and test (modified in place)
#         # --- by default replacing ALL indexes of class
#         replace_class(train_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
#         if val:
#             replace_class(valid_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
#         # replace_class(test_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)

#         # if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
#         #     test_set.data = test_set.data[test_set.targets != class_to_replace]
#         #     test_set.targets = test_set.targets[test_set.targets != class_to_replace]

#     if percent_to_replace is not None:
#         # replace some percentage across all classes (modified in place)
#         replace_random_uniform(train_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
#         if val:
#             replace_random_uniform(valid_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
#         # replace_random_uniform(test_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)

#     loader_args = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         **loader_args,
#     )

#     if val:
#         val_loader = DataLoader(
#             valid_set,
#             batch_size=batch_size,
#             shuffle=False,
#             **loader_args,
#         )
#     else:
#         val_loader = None

#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         **loader_args,
#     )

#     return train_loader, val_loader, test_loader




def generic_dataloaders(
    train_set, 
    test_set, 
    replace_type,
    value_to_replace, # a dictionary of one or more values
    batch_size=128,
    num_workers=0,
    seed: int = 1,
    only_mark: bool = False,
    val = True):


    # access labels/targets, and just call them "targets", ...
    train_set.targets = get_targets(train_set)
    test_set.targets = get_targets(test_set)
    # sync .labels for datasets like SVHN whose __getitem__ reads .labels not .targets
    if hasattr(train_set, 'labels'):
        train_set.labels = train_set.targets
    if hasattr(test_set, 'labels'):
        test_set.labels = test_set.targets

    # IF WE NEED A VALIDATION SET, SPLIT IT OFF FROM THE TRAINING SET (10% of each class)
    # and then reconfigure train set to accommodate
    if val:
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.data = train_set_copy.data[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]
        if hasattr(valid_set, 'labels'):
            valid_set.labels = valid_set.targets

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.data = train_set_copy.data[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]
        if hasattr(train_set, 'labels'):
            train_set.labels = train_set.targets


    # --- carry out the replacement, whether class, random, random_uniform, etc
    replace_func = get_replace_type(replace_type)

    replace_type_seeds = {
        "class": 1,
        "random": 2,
        "poisoned": 3,
        "random_uniform": 4
    }

    replace_seed = replace_type_seeds[replace_type]
    replace_func(train_set, value_to_replace, seed=replace_seed, only_mark=only_mark)
    if val:
        replace_func(valid_set, value_to_replace, seed=replace_seed, only_mark=only_mark)

    loader_args = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        **loader_args,
    )

    if val:
        val_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            **loader_args,
        )
    else:
        val_loader = None

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def get_dataloader_method(name = "CIFAR10"):
    
    if name == "CIFAR10":
        return cifar10_dataloaders
    elif name == "SVHN":
        return svhn_dataloaders
    else:
        raise NotImplementedError(f"Data loaders for {name} not implemented!")




def cifar10_dataloaders(
    data_dir="data/datasets/cifar10",
    replace_type: int = None,
    value_to_replace=None,
    batch_size=128,
    num_workers=0,
    seed: int = 1,
    only_mark: bool = False,
    val = True
):
    
    # --- set transforms for both train and test splits --- #
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(degrees = 10),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2023, 0.1994, 0.2010]), 
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]), 
        ]
    )

    # --- load the dataset --- #
    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    
    # --- make loaders --- #
    # train_loader, val_loader, test_loader = generic_dataloaders(train_set, test_set, batch_size, num_workers, class_to_replace, percent_to_replace, seed, only_mark, val)
    train_loader, val_loader, test_loader = generic_dataloaders(
        train_set, 
        test_set, 
        replace_type = replace_type, 
        value_to_replace = value_to_replace, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        seed = seed, 
        only_mark = only_mark, 
        val = val)

    # --- print what we did --- #
    print("="*10 + " DATALOADER INFO")
    print("Dataset: CIFAR-10")
    print(f"Train: {len(train_set)} images for training")
    if val:
        print(f"Val: {len(val_loader.dataset)} images for validation")
    print(f"Test: {len(test_set)} images for testing")
    # if class_to_replace is not None:
    #     print(f"Replaced class {class_to_replace} in train")
    # if percent_to_replace is not None:
    #     print(f"Replaced {percent_to_replace:.1f}% of labels across all classes in train")
    print(f"Replace type = {replace_type}, value to replace = {value_to_replace}")
    print("Training augmentation = randomcrop(32,4) + randomhorizontalflip + colorjitter + randomrotation + normalize")
    print("Validation/Test augmentation = normalize")
    print(f"num_workers = {num_workers}")
    print("\n")

    return train_loader, val_loader, test_loader




# def svhn_dataloaders(
#     batch_size=128,
#     data_dir="datasets/svhn",
#     num_workers=2,
#     class_to_replace: int = None,
#     num_indexes_to_replace=None,
#     indexes_to_replace=None,
#     seed: int = 1,
#     only_mark: bool = False,
#     shuffle=True,
#     no_aug=False,
# ):
#     train_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     test_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     print(
#         "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
#     )

#     train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

#     test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

#     train_set.labels = np.array(train_set.labels)
#     test_set.labels = np.array(test_set.labels)

#     rng = np.random.RandomState(seed)
#     valid_set = copy.deepcopy(train_set)
#     valid_idx = []
#     for i in range(max(train_set.labels) + 1):
#         class_idx = np.where(train_set.labels == i)[0]
#         valid_idx.append(
#             rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
#         )
#     valid_idx = np.hstack(valid_idx)
#     train_set_copy = copy.deepcopy(train_set)

#     valid_set.data = train_set_copy.data[valid_idx]
#     valid_set.labels = train_set_copy.labels[valid_idx]

#     train_idx = list(set(range(len(train_set))) - set(valid_idx))

#     train_set.data = train_set_copy.data[train_idx]
#     train_set.labels = train_set_copy.labels[train_idx]

#     if class_to_replace is not None and indexes_to_replace is not None:
#         raise ValueError(
#             "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
#         )
#     if class_to_replace is not None:
#         replace_class(
#             train_set,
#             class_to_replace,
#             num_indexes_to_replace=num_indexes_to_replace,
#             seed=seed - 1,
#             only_mark=only_mark,
#         )
#         if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
#             test_set.data = test_set.data[test_set.labels != class_to_replace]
#             test_set.labels = test_set.labels[test_set.labels != class_to_replace]

#     if indexes_to_replace is not None:
#         replace_indexes(
#             dataset=train_set,
#             indexes=indexes_to_replace,
#             seed=seed - 1,
#             only_mark=only_mark,
#         )

#     loader_args = {"num_workers": 0, "pin_memory": False}

#     def _init_fn(worker_id):
#         np.random.seed(int(seed))

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         worker_init_fn=_init_fn if seed is not None else None,
#         **loader_args,
#     )
#     val_loader = DataLoader(
#         valid_set,
#         batch_size=batch_size,
#         shuffle=False,
#         worker_init_fn=_init_fn if seed is not None else None,
#         **loader_args,
#     )
#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         worker_init_fn=_init_fn if seed is not None else None,
#         **loader_args,
#     )

#     return train_loader, val_loader, test_loader




def svhn_dataloaders(
    data_dir="data/datasets/svhn",
    replace_type: int = None,
    value_to_replace=None,
    batch_size=128,
    num_workers=2,
    seed: int = 1,
    only_mark: bool = False,
    val = True
):
    # set transforms
    train_transform = transforms.Compose([transforms.ToTensor(),])

    test_transform = transforms.Compose([transforms.ToTensor(),])

    # load datasets
    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)
    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    # get data loaders
    # train_loader, val_loader, test_loader = generic_dataloaders(train_set, test_set, batch_size, num_workers, class_to_replace, percent_to_replace, seed, only_mark, val)
    train_loader, val_loader, test_loader = generic_dataloaders(
        train_set, 
        test_set, 
        replace_type = replace_type, 
        value_to_replace = value_to_replace, 
        batch_size = batch_size, 
        num_workers = num_workers, 
        seed = seed, 
        only_mark = only_mark, 
        val = val)

    # print what we did
    # print("Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t")
    print("="*10 + " DATALOADER INFO")
    print("Dataset: SVHN")
    print(f"Train: {len(train_set)} images for training")
    if val:
        print(f"Val: {len(val_loader.dataset)} images for validation")
    print(f"Test: {len(test_set)} images for testing")
    # if class_to_replace is not None:
    #     print(f"Replaced class {class_to_replace} in train")
    # if percent_to_replace is not None:
    #     print(f"Replaced {percent_to_replace:.1f}% of labels across all classes in train")
    print(f"Replace type = {replace_type}, value to replace = {value_to_replace}")
    print("Training augmentation = None")
    print("Validation/Test augmentation = None")
    print(f"num_workers = {num_workers}")
    print("\n")

    return train_loader, val_loader, test_loader


def load_dataloaders_for_experiment(
    name,
    replace_type = None,
    value_to_replace=None,
    batch_size=128,
    num_workers=2,
    seed: int = 1,
    only_mark: bool = False,
    val = True
    ):

    dataloader_func = get_dataloader_method(name)

    train_loader, val_loader, test_loader = dataloader_func(
        replace_type = replace_type,
        value_to_replace=value_to_replace,
        batch_size=batch_size,
        num_workers=num_workers,
        seed = seed,
        only_mark = only_mark,
        val = val
    )

    return train_loader, val_loader, test_loader





if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_dataloaders_for_experiment(name = "CIFAR10")
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)


