import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# class NTK_Linear(nn.Module):

#     def __init__(self, input_dim, output_dim):

#         super(NTK_Linear, self).__init__() 
#         # Calling Super Class's constructor
#         self.linear = nn.Linear(input_dim, output_dim,bias=False)
#         # nn.linear is defined in nn.Module

#     def forward(self, x):
#         # Here the forward pass is simply a linear function

#         out = self.linear(x)
#         return out

# class LinearNeuralTangentKernel(nn.Linear): 
    
#     def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
#         self.beta = beta
#         super(LinearNeuralTangentKernel, self).__init__(in_features, out_features)
#         self.reset_parameters()
#         self.w_sig = w_sig
      
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, mean=0, std=1)
#         if self.bias is not None:
#             torch.nn.init.normal_(self.bias, mean=0, std=1)

#     def forward(self, input):
#         return F.linear(input, self.w_sig * self.weight/np.sqrt(self.in_features), self.beta * self.bias)

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}, beta={}'.format(
#             self.in_features, self.out_features, self.bias is not None, self.beta)

# class NTK_MLP(nn.Module):
#     def __init__(self, num_classes=10, filters_percentage=1.0, beta=np.sqrt(0.1)):
#         super(NTK_MLP, self).__init__()
#         self.n_wid = int(32*filters_percentage)
#         self.fc1 = LinearNeuralTangentKernel(1024, self.n_wid, beta=beta)
#         self.fc2 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)
# #         self.fc3 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
# #         self.fc4 = LinearNeuralTangentKernel(self.n_wid, self.n_wid, beta=beta)
# #         self.fc5 = LinearNeuralTangentKernel(self.n_wid, num_classes, beta=beta)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = F.relu(self.fc3(x))
# #         x = F.relu(self.fc4(x))
#         x = self.fc2(x)
#         return x

# class Affine(nn.Module):

#     def __init__(self, num_features):
#         super().__init__()
#         self.weight = Parameter(torch.Tensor(num_features))
#         self.bias = Parameter(torch.Tensor(num_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         init.ones_(self.weight)
#         init.zeros_(self.bias)

#     def forward(self, x):
#         return x * self.weight + self.bias
    
# class StandardLinearLayer(nn.Linear): 
    
#     def __init__(self, in_features, out_features, bias=True, beta=np.sqrt(0.1), w_sig = np.sqrt(2.0)):
#         self.beta = beta
#         self.w_sig = w_sig
#         super(StandardLinearLayer, self).__init__(in_features, out_features)
#         self.reset_parameters()
      
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/np.sqrt(self.in_features))
#         if self.bias is not None:
#             torch.nn.init.normal_(self.bias, mean=0, std=self.beta)

#     def forward(self, input):
#         return F.linear(input, self.weight, self.bias)

#     def extra_repr(self):
#         return 'in_features={}, out_features={}, bias={}, beta={}'.format(
#             self.in_features, self.out_features, self.bias is not None, self.beta)
    
# class MLP(nn.Module):

#     def __init__(self, num_layer=1, num_classes=10, filters_percentage=1., hidden_size=32, input_size=1024):
#         super(MLP, self).__init__()
#         self.input_size = input_size
#         self.num_layer = num_layer
#         self.num_classes = num_classes
#         self.hidden_size = hidden_size
#         self.layers = self._make_layers()

#     def _make_layers(self):
#         layer = []
#         layer += [
#             StandardLinearLayer(self.input_size,self.hidden_size),#nn.Linear(self.input_size, self.hidden_size),
#             # Affine(self.hidden_size),
#             nn.ReLU()]
#         for i in range(self.num_layer - 2):
#             layer += [StandardLinearLayer(self.hidden_size,self.hidden_size)]#[nn.Linear(self.hidden_size, self.hidden_size)]
#             # layer += [Affine(self.hidden_size)]
#             layer += [nn.ReLU()]
#         layer += [StandardLinearLayer(self.hidden_size,self.num_classes)]#[nn.Linear(self.hidden_size, self.num_classes)]
#         return nn.Sequential(*layer)

#     def forward(self, x):
#         x = x.reshape(x.size(0), self.input_size)
#         return self.layers(x)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)
    
# class ConvStandard(nn.Conv2d): 
    
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
#                  np.sqrt(1.0)):
#         super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         self.kernel_size=kernel_size
#         self.stride=stride
#         self.padding=padding
#         self.w_sig = w_sig
#         self.reset_parameters()
      
#     def reset_parameters(self):
#         torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
#         if self.bias is not None:
#             torch.nn.init.normal_(self.bias, mean=0, std=0)
            
#     def forward(self, input):
#         return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else nn.Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else nn.Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
