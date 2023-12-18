import torch
import torch.nn as nn
from torchvision import models

# First part: ResNet with an additional layer to transform to 384 features
class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fc layer.
        self.conv = nn.Conv2d(2048, 384, 1)  # Additional conv layer to get 384 features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        return x

class UltraLightDecoder(nn.Module):
    def __init__(self):
        super(UltraLightDecoder, self).__init__()
        # Single Convolutional layer to start reducing the channel dimensions
        self.reduce_conv = nn.Conv2d(384, 16, kernel_size=1)  # 1x1 Conv to reduce channels

        # Upsample and then apply a final convolution
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=224, mode='nearest'),  # Large upsampling
            nn.Conv2d(16, 3, kernel_size=3, padding=1)     # Final layer to get 3 channels
        )

    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.upconv(x)
        return x

# Combine both parts
class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.encoder = ResNetEncoder()
        self.decoder = UltraLightDecoder()

        num_encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print("params encoder", num_encoder_params)

        num_ultra_light_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        print("params decoder", num_ultra_light_params)

    def forward(self, x):
        enc = self.encoder(x)
        #print(enc.shape)
        x = self.decoder(enc)
        #print(x.shape)
        return x, enc.flatten(1)

"""
# Instantiate the model
model = MyResNet()

# Dummy input
input_tensor = torch.randn(1, 3, 224, 224)

# Forward pass
output_tensor, enc = model(input_tensor)
print(output_tensor.shape) # Should be torch.Size([1, 3, 224, 224])
print(enc.shape)
"""
