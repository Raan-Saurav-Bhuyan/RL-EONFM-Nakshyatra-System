import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

from .model import ContrastiveModel
from .augmentations import Augmentation

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    def __init__(self, temperature=0.5, device='cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """
        Calculates the NT-Xent loss for a batch of positive pairs (z_i, z_j).
        """
        N = len(z_i)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * N, 1)

        mask = (~torch.eye(2 * N, 2 * N, dtype = bool)).to(self.device)
        negative_samples = sim[mask].reshape(2 * N, -1)

        labels = torch.zeros(2 * N).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim = 1)

        loss = self.criterion(logits, labels)
        loss /= (2 * N)

        return loss

class Trainer:
    """Handles the training process for the Self-Supervised Contrastive model."""
    def __init__(self, input_dim, representation_dim = 32, projection_dim = 16,
                 epochs = 100, lr = 1e-3, batch_size = 256, temperature = 0.5,
                 seed = 42, verbose = False):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ContrastiveModel(input_dim, representation_dim, projection_dim).to(self.device)
        self.augmenter = Augmentation(device=self.device)
        self.loss_fn = NTXentLoss(temperature=temperature, device=self.device)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.verbose = verbose

    def train(self, X):
        """Trains the contrastive model."""
        if self.verbose:
            print(f"... Training Contrastive Model on {self.device}")

        effective_batch_size = min(self.batch_size, X.shape[0])
        dataset = TensorDataset(torch.from_numpy(X).float())
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data in dataloader:
                x = batch_data[0].to(self.device)

                x_i, x_j = self.augmenter(x)

                z_i = F.normalize(self.model(x_i), p=2, dim=1)
                z_j = F.normalize(self.model(x_j), p=2, dim=1)

                loss = self.loss_fn(z_i, z_j)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

        if self.verbose:
            print("... Contrastive training complete.")
