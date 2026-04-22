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


# def cifar10_dataloaders_no_val(
#     batch_size=128, data_dir="datasets/cifar10", num_workers=2
# ):
#     train_transform = transforms.Compose(
#         [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#         ]
#     )

#     test_transform = transforms.Compose(
#         [
#             transforms.ToTensor(),
#         ]
#     )

#     print(
#         "Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t"
#     )
#     print("10000 images for testing\t no normalize applied in data_transform")
#     print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

#     train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
#     val_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=False,
#     )
#     val_loader = DataLoader(
#         val_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=False,
#     )
#     test_loader = DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=False,
#     )

#     return train_loader, val_loader, test_loader


def svhn_dataloaders(
    batch_size=128,
    data_dir="datasets/svhn",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: SVHN\t 45000 images for training \t 5000 images for validation\t"
    )

    train_set = SVHN(data_dir, split="train", transform=train_transform, download=True)

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_set.labels = np.array(train_set.labels)
    test_set.labels = np.array(test_set.labels)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.labels) + 1):
        class_idx = np.where(train_set.labels == i)[0]
        valid_idx.append(
            rng.choice(class_idx, int(0.1 * len(class_idx)), replace=False)
        )
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.labels = train_set_copy.labels[valid_idx]

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.labels = train_set_copy.labels[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None or num_indexes_to_replace == 4454:
            test_set.data = test_set.data[test_set.labels != class_to_replace]
            test_set.labels = test_set.labels[test_set.labels != class_to_replace]

    if indexes_to_replace is not None:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar100",
    num_workers=2,
    class_to_replace: int = None,
    num_indexes_to_replace=None,
    indexes_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
    no_aug=False,
):
    if no_aug:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")
    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

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

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
        )
    if class_to_replace is not None:
        replace_class(
            train_set,
            class_to_replace,
            num_indexes_to_replace=num_indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets != class_to_replace]
    if indexes_to_replace is not None or indexes_to_replace == 450:
        replace_indexes(
            dataset=train_set,
            indexes=indexes_to_replace,
            seed=seed - 1,
            only_mark=only_mark,
        )

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(
    batch_size=128, data_dir="datasets/cifar100", num_workers=2
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    print(
        "Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t"
    )
    print("10000 images for testing\t no normalize applied in data_transform")
    print("Data augmentation = randomcrop(32,4) + randomhorizontalflip")

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start:end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
    TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = (
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else None
        )

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, "train/")
        self.val_path = os.path.join(args.data_dir, "val/")
        self.test_path = os.path.join(args.data_dir, "test/")

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, "r") as f:
                for line in f.readlines():
                    split_line = line.split("\t")
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, "val/images/*"))
            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + "/images")
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + "/images")

            for path in paths:
                file = path.split("/")[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + "/images/*")) < 25:
                    dest = self.val_path + str(folder) + "/images/" + str(file)
                else:
                    dest = self.test_path + str(folder) + "/images/" + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(
        self,
        batch_size=128,
        data_dir="datasets/tiny",
        num_workers=2,
        class_to_replace: int = None,
        num_indexes_to_replace=None,
        indexes_to_replace=None,
        seed: int = 1,
        only_mark: bool = False,
        shuffle=True,
        no_aug=False,
    ):
        train_set = ImageFolder(self.train_path, transform=self.tr_train)
        train_set = TinyImageNetDataset(train_set, self.norm_layer)
        test_set = ImageFolder(self.test_path, transform=self.tr_test)
        test_set = TinyImageNetDataset(test_set, self.norm_layer)
        train_set.targets = np.array(train_set.targets)
        train_set.targets = np.array(train_set.targets)
        rng = np.random.RandomState(seed)
        valid_set = copy.deepcopy(train_set)
        valid_idx = []
        for i in range(max(train_set.targets) + 1):
            class_idx = np.where(train_set.targets == i)[0]
            valid_idx.append(
                rng.choice(class_idx, int(0.0 * len(class_idx)), replace=False)
            )
        valid_idx = np.hstack(valid_idx)
        train_set_copy = copy.deepcopy(train_set)

        valid_set.imgs = train_set_copy.imgs[valid_idx]
        valid_set.targets = train_set_copy.targets[valid_idx]

        train_idx = list(set(range(len(train_set))) - set(valid_idx))

        train_set.imgs = train_set_copy.imgs[train_idx]
        train_set.targets = train_set_copy.targets[train_idx]

        if class_to_replace is not None and indexes_to_replace is not None:
            raise ValueError(
                "Only one of `class_to_replace` and `indexes_to_replace` can be specified"
            )
        if class_to_replace is not None:
            replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )
            if num_indexes_to_replace is None or num_indexes_to_replace == 500:
                test_set.targets = np.array(test_set.targets)
                test_set.imgs = test_set.imgs[test_set.targets != class_to_replace]
                test_set.targets = test_set.targets[
                    test_set.targets != class_to_replace
                ]
                test_set.targets = test_set.targets.tolist()
        if indexes_to_replace is not None:
            replace_indexes(
                dataset=train_set,
                indexes=indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

        loader_args = {"num_workers": 0, "pin_memory": False}

        def _init_fn(worker_id):
            np.random.seed(int(seed))

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        val_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            worker_init_fn=_init_fn if seed is not None else None,
            **loader_args,
        )
        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


def cifar10_dataloaders(
    batch_size=128,
    data_dir="datasets/cifar10",
    num_workers=0,
    random_to_replace: int = None,
    class_to_replace: int = None,
    percent_to_replace=None,
    seed: int = 1,
    only_mark: bool = False,
    shuffle=True,
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
    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)


    # --- pick random 10% (within each class) of the training set to be the validation set --- #
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

    train_idx = list(set(range(len(train_set))) - set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    # --- replace some data if specified --- #

    # ------ cant specify both class and number to replace
    if class_to_replace is not None and percent_to_replace is not None:
        raise ValueError("Only one of `class_to_replace` and `percent_to_replace` can be specified")


    if class_to_replace is not None:
        
        # replace classes in training, val, and test (modified in place)
        # --- by default replacing ALL indexes of class
        replace_class(train_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
        replace_class(valid_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)
        replace_class(test_set, class_to_replace, seed=int(f"{seed}{class_to_replace}"), only_mark=only_mark)

        # if num_indexes_to_replace is None or num_indexes_to_replace == 4500:
        #     test_set.data = test_set.data[test_set.targets != class_to_replace]
        #     test_set.targets = test_set.targets[test_set.targets != class_to_replace]

    if percent_to_replace is not None:
        # replace some percentage across all classes (modified in place)
        replace_percent(train_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
        replace_percent(valid_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)
        replace_percent(test_set, percent_to_replace, seed=int(f"{seed}{percent_to_replace}"), only_mark=only_mark)

    loader_args = {"num_workers": 0, "pin_memory": False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    val_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        worker_init_fn=_init_fn if seed is not None else None,
        **loader_args,
    )

    # --- print what we did --- #

    print("="*10 + " DATALOADER INFO")
    print("Dataset: CIFAR-10")
    print(f"Train: {len(train_set)} images for training")
    print(f"Val: {len(valid_set)} images for validation")
    print(f"Test: {len(test_set)} images for testing")
    if class_to_replace is not None:
        print(f"Replaced class {class_to_replace} in train, val, and test")
    if percent_to_replace is not None:
        print(f"Replaced {percent_to_replace:.1f}% of labels across all classes in train, val, and test")
    print("Training augmentation = randomcrop(32,4) + randomhorizontalflip + colorjitter + randomrotation + normalize")
    print("Validation/Test augmentation = normalize")
    print("\n")

    return train_loader, val_loader, test_loader


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
            dataset.labels[indexes] = dataset.labels[new_indexes]
        else:
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


def replace_percent(
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
        num_to_replace = int(len(class_indexes) * (percent_to_replace*.01))
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
              f"({percent_to_replace:.1f}% across all {len(unique_classes)} classes)")
        
        replace_indexes(dataset, all_selected_indexes, seed, only_mark)
    else:
        print("Warning: Percentage too low or dataset too small to replace any samples.")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
