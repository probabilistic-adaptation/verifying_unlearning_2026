import copy
import os
import time
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate
import pickle

def calculate_outputs(model, loader, device):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
    return torch.cat(all_outputs, dim=0)  

def evaluation_callback(unlearn_data_loaders, epoch, model, args, criterion, unlearn_time_ms,**kwargs):
    device = kwargs.get('device')
    original_model = kwargs.get('original_model')
    original_forget_outputs = kwargs.get('original_forget_outputs')
    forget_loader = kwargs.get('forget_loader')
    evaluation_result = {}
    
    # Save model snapshot
    checkpoint_path = os.path.join(args.save_dir, f'model_epoch_{epoch}.pth.tar')
    torch.save({'state_dict': model.state_dict()}, checkpoint_path)
    print(f"Saved model snapshot at epoch {epoch}")
    
    # Calculate unlearned forget outputs
    unlearned_forget_outputs = calculate_outputs(model, forget_loader, device)
    
    # Evaluation metrics
    
    # Metric 2: Wasserstein distance
    wasserstein_dist = evaluation.get_wasserstein_distance(
        original_forget_outputs, unlearned_forget_outputs
    )
    evaluation_result["wasserstein_dist"] = wasserstein_dist
    
    # Metric 3: Activation distance
    activ_dist_forget = evaluation.get_activation_distance(
        model1=original_model,
        model2=model,
        dataloader=forget_loader,
        device=device
    )
    evaluation_result["activ_dist_forget"] = activ_dist_forget

    # Metrics 3 and 4 between unlearned model and retrained model
    if args.unlearn == 'retrain':
        # For retrain method, set distances to 0
        evaluation_result['wasserstein_dist_retrain'] = 0
        evaluation_result['activ_dist_retrain'] = 0
    else:
        retrain_model = kwargs.get('retrain_model')
        retrain_forget_outputs = kwargs.get('retrain_forget_outputs')

        wasserstein_dist_retrain = evaluation.get_wasserstein_distance(
            retrain_forget_outputs, unlearned_forget_outputs
        )
        evaluation_result['wasserstein_dist_retrain'] = wasserstein_dist_retrain

        activ_dist_retrain = evaluation.get_activation_distance(
            model1=retrain_model,
            model2=model,
            dataloader=forget_loader,
            device=device
        )
        evaluation_result['activ_dist_retrain'] = activ_dist_retrain

    
    # Metric 4: Accuracy on different datasets
    accuracy = {}
    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args)
        accuracy[name] = val_acc
        print(f"{name} acc: {val_acc}")
    
    evaluation_result["accuracy"] = accuracy
    
    # Remove deprecated metrics
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)
    
    # Compute SVC_MIA_forget_efficacy if not already present
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        # Retrieve required datasets
        test_loader = unlearn_data_loaders.get('test')
        retain_loader = unlearn_data_loaders.get('retain')
        
        # Ensure datasets are available
        if test_loader is None or retain_loader is None:
            print("Error: test_loader or retain_loader is not available for SVC_MIA computation.")
        else:
            # Get datasets from loaders
            test_dataset = test_loader.dataset
            retain_dataset = retain_loader.dataset
            forget_dataset = forget_loader.dataset  # Already available
            
            test_len = len(test_dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)
            
            # Convert datasets to test mode
            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_dataset, args)
            utils.dataset_convert_to_test(test_dataset, args)
            
            # Create shadow training data
            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )
            
            # Compute SVC_MIA_forget_efficacy
            evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=forget_loader,
                model=model,
            )

    evaluation_result['unlearn_time_ms'] = unlearn_time_ms


    # Save evaluation result
    evaluation_result_path = os.path.join(args.save_dir, f'evaluation_epoch_{epoch}.pkl')
    with open(evaluation_result_path, 'wb') as f:
        pickle.dump(evaluation_result, f)
    print(f"Saved evaluation results at epoch {epoch}")

def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()
    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        #loard model and capture original activations
        checkpoint = torch.load(args.model_path, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        #Original model weights and outptus(preds)
        original_model= copy.deepcopy(model)
        original_forget_outputs = calculate_outputs(model, forget_loader, device)

        additional_kwargs = {
            'callback': evaluation_callback,
            'original_model': original_model,
            'original_forget_outputs': original_forget_outputs,
            'forget_loader': forget_loader,
            'device': device,
        }

        if args.unlearn != 'retrain':
            unlearn_model_lower = args.unlearn.lower()
            retrain_epochs = 10
            retrain_model_path = os.path.join(
                args.save_dir.replace(unlearn_model_lower, 'retrain'),
                #TODO: Change retrain_epochs to args.retrain_epochs
                f'model_epoch_{retrain_epochs}.pth.tar'
            )
            retrain_model = copy.deepcopy(model)
            checkpoint = torch.load(retrain_model_path, map_location=device)
            if 'state_dict' in checkpoint:
                retrain_model.load_state_dict(checkpoint['state_dict'])
            else:
                retrain_model.load_state_dict(checkpoint)
            retrain_model = retrain_model.to(device)
            retrain_model.eval()
            retrain_forget_outputs = calculate_outputs(retrain_model, forget_loader, device)

            additional_kwargs['retrain_model'] = retrain_model
            additional_kwargs['retrain_forget_outputs'] = retrain_forget_outputs

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_method(unlearn_data_loaders, model, criterion, args, **additional_kwargs)


    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()