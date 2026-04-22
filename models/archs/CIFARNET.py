
import torch
import torch.nn as nn

class CIFARNET(nn.Module):
    def __init__(self, input_dims = 3, frame_size = 32):
        """
        """
        super().__init__()
        self.input_dims = input_dims
        self.frame_size = frame_size

        # 32 x 32 -> 8 x 8
        self.conv = nn.Sequential(

            
            nn.Conv2d(self.input_dims, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2) ),
            nn.Dropout(.25),

            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d( (2, 2)),
            nn.Dropout(.25),

        )

        self.flatten = nn.Flatten() # some dim, depending on input_dim
        
        # the size of the cnn output
        with torch.no_grad():
            dummy_in = torch.zeros(1, self.input_dims, self.frame_size, self.frame_size)
            dummy_out = self.conv(dummy_in)
            n_flat = self.flatten(dummy_out).shape[1]

        # This will be the first hidden layer
        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(512, 10)
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