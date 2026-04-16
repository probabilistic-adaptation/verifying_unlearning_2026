
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_dims = 3, frame_size = 32):
        """
        """
        super().__init__()
        self.input_dims = input_dims
        self.frame_size = frame_size

        self.conv = nn.Sequential(

            # 28x28 -> 14x14
            nn.Conv2d(self.input_dims, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2) ),
            nn.Dropout(.2),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2)),
            nn.Dropout(.2),

            # 7x7 -> 4x4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2) ),
            nn.Dropout(.2),
            
            # 4x4 -> 2x2
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.AdaptiveMaxPool2d(1)
        )

        self.flatten = nn.Flatten() # some dim, depending on input_dim
        
        # the size of the cnn output
        with torch.no_grad():
            dummy_in = torch.zeros(1, self.input_dims, self.frame_size, self.frame_size)
            dummy_out = self.conv(dummy_in)
            n_flat = self.flatten(dummy_out).shape[1]

        # This will be the first hidden layer
        self.head = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.Dropout(.3),
            nn.ReLU(),
            nn.Linear(128, 10)
            )
        
    def forward(self, X):
        """
        The actual forward computation is defined here. This is run whenever you "call" the
        `nn.Module`, after it is instantiated.
        
        Here we will write down the computation required to run the model forward, calling
        our previously defined subnetworks
        
        """

        x = self.conv(X)
        x = self.flatten(x)
        y = self.head(x)
        return y