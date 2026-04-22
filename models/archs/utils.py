import torch
from models.archs.ConvNet import ConvNet
from models.archs.VGG import vgg11_bn
from models.archs.CIFARNET import CIFARNET
from models.archs.ResNets import resnet20s




def init_model(model_class, checkpoint_path=None):

    if model_class == "ConvNet":

        model = ConvNet()

    elif model_class == "VGG":
        model = vgg11_bn(pretrained=False)

    elif model_class == "CIFARNET":
        model = CIFARNET()
    elif model_class == "ResNet":
        model = resnet20s()

    else:
        raise NotImplementedError(f"{model_class} is not an implemented model_class")
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model

