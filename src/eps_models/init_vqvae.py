import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_x = x.permute(0, 2, 3, 1).contiguous()
        flat_x = flat_x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = F.embedding(encoding_indices, self.embedding.weight).view(flat_x.shape)

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_x)
        q_latent_loss = F.mse_loss(quantized, flat_x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = flat_x + (quantized - flat_x).detach()
        quantized = quantized.view(x.shape)

        return loss, quantized

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, output_channels):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=8, mode='nearest')
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(input_channels, hidden_channels, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, input_channels)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, quantized.flatten(1), loss

# Example usage
# vqvae = VQVAE(input_channels=3, hidden_channels=16, embedding_dim=384, num_embeddings=64, commitment_cost=0.25)
# image = torch.randn(32, 3, 32, 32)  # Example image tensor
# x_recon, quantized, loss = vqvae(image)

# print(x_recon.shape)
# print(quantized.shape)

# num_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
# print(num_params)