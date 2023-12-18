import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        ### some conv layers ###
        # 224 x 224 x 3 -> 112 x 112 x 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 112 x 112 x 64 -> 56 x 56 x 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 56 x 56 x 128 -> 28 x 28 x 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # 28 x 28 x 256 -> 14 x 14 x 512
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # 14 x 14 x 512 -> 7 x 7 x 512
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        ### some linear layer ###
        # compress to 384
        self.compress = nn.Linear(7 * 7 * 512, 384)

        ### some linear layer ###
        # upscale and reshape to 224 x 224
        self.upscale = nn.Linear(384, 224 * 224 * 3)

    def forward(self, x):
        ### some conv layers ###
        # 224 x 224 x 3 -> 112 x 112 x 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # 112 x 112 x 64 -> 56 x 56 x 128
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # 56 x 56 x 128 -> 28 x 28 x 256
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # 28 x 28 x 256 -> 14 x 14 x 512
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # 14 x 14 x 512 -> 7 x 7 x 512
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        ### compress ###
        x = x.view(-1, 7 * 7 * 512)
        x = self.compress(x)
        encoding = F.relu(x)

        ### upscale and reshape ###
        x = self.upscale(encoding)
        x = x.view(-1, 3, 224, 224)
        return x, encoding

### Testing ###
#model = ConvNetConditioner()
#x = torch.randn(200, 3, 224, 224)
#output, encoding = model(x)
#
#print(output.shape)
#print(encoding.shape)

