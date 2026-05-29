import torch
from models.archs.ConvNet import ConvNet
from models.archs.VGG import vgg11_bn
from models.archs.CIFARNET import CIFARNET
from models.archs.ResNet import resnet18
from models.archs.AllCNN import AllCNN



def init_model(model_class, checkpoint_path=None):

    if model_class == "ConvNet":

        model = ConvNet()

    elif model_class == "VGG":
        model = vgg11_bn(pretrained=False)

    elif model_class == "CIFARNET":
        model = CIFARNET()
    elif model_class == "ResNet":
        model = resnet18()
    elif model_class == "AllCNN":
        model = AllCNN()

    else:
        raise NotImplementedError(f"{model_class} is not an implemented model_class")
    
    if checkpoint_path is not None:
        
        # Load whatever is in the file
        checkpoint = torch.load(checkpoint_path)
        
        # Check if it was saved as a nested checkpoint dictionary
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            
        # Otherwise, assume it's a raw state dictionary
        else:
            model.load_state_dict(checkpoint)

    return model

