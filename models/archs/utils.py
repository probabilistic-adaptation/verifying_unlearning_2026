import torch
from models.archs.ConvNet import ConvNet
from models.archs.VGG import vgg11_bn
from models.archs.CIFARNET import CIFARNET
from models.archs.ResNet import resnet18
from models.archs.AllCNN import AllCNN
from models.archs.ViT import ViT



def init_model(model_class, num_classes = None, checkpoint_path=None, device = "cpu"):

    if model_class == "ConvNet":
        model = ConvNet()

    elif model_class == "VGG":
        model = vgg11_bn(pretrained=False)

    elif model_class == "CIFARNET":
        model = CIFARNET()
    elif model_class == "ResNet":
        model = resnet18(num_classes = num_classes)
    elif model_class == "AllCNN":
        model = AllCNN()
    elif model_class == "ViT":
        model = ViT(num_classes=num_classes)

    else:
        raise NotImplementedError(f"{model_class} is not an implemented model_class")
    
    if checkpoint_path is not None:
        
        # Load whatever is in the file
        checkpoint = torch.load(checkpoint_path, map_location = torch.device(device))
        
        # Check if it was saved as a nested checkpoint dictionary
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            
        # Otherwise, assume it's a raw state dictionary
        else:
            model.load_state_dict(checkpoint)

    return model

