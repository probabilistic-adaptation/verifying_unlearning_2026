import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pool=True):
        super().__init__()
        
        # 1. The Main Convolutional Pathway
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # We handle pooling separately so we can apply it to the shortcut too
        self.pool = nn.MaxPool2d((2, 2)) if apply_pool else nn.Identity()
        
        # 2. The Shortcut Pathway (Projection)
        # If input and output channels differ, use a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Pass data through the main convolutions
        out = self.main_path(x)
        
        # Pass original data through the shortcut 1x1 conv
        skip = self.shortcut(x)
        
        # Apply pooling to BOTH pathways so their spatial dimensions match
        out = self.pool(out)
        skip = self.pool(skip)
        
        # Add them together and apply final activation
        return nn.functional.relu(out + skip)


class OtherConvNet(nn.Module):
    def __init__(self, input_dims=3, frame_size=32):
        super().__init__()
        self.input_dims = input_dims
        self.frame_size = frame_size

        # Replace the massive nn.Sequential with our new ResBlocks
        self.layer1 = ResBlock(self.input_dims, 64, apply_pool=True)  # -> 16x16
        self.layer2 = ResBlock(64, 128, apply_pool=True)              # -> 8x8
        self.layer3 = ResBlock(128, 256, apply_pool=True)             # -> 4x4
        self.layer4 = ResBlock(256, 512, apply_pool=False)            # -> 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # Calculate the size of the CNN output dynamically
        with torch.no_grad():
            dummy_in = torch.zeros(1, self.input_dims, self.frame_size, self.frame_size)
            x = self.layer1(dummy_in)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            dummy_out = self.flatten(x)
            n_flat = dummy_out.shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.Dropout(.3),
            nn.ReLU(),
            nn.Linear(256, 10) 
        )
        
    def forward(self, X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        y = self.head(x)
        return y