import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import numpy as np

from .model import Autoencoder, ClusteringLayer

def target_distribution(q):
    """
    Computes the target distribution p_ij by raising q_ij to the second power
    and normalizing by frequency, as described in the paper.
    """
    weight = q**2 / torch.sum(q, 0)

    return (weight.t() / torch.sum(weight, 1)).t()

class Trainer:
    """
    Handles the training process for the unsupervised similarity learning model.

    This includes pre-training the autoencoder and jointly training the
    autoencoder with the clustering layer.
    """
    def __init__(self, input_dim, n_clusters, latent_dim = 10, pretrain_epochs = 50,
                 train_epochs = 100, lr = 1e-3, batch_size = 256, seed = 42, verbose = False):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = Autoencoder(input_dim, latent_dim).to(self.device)
        self.clustering_layer = ClusteringLayer(n_clusters, latent_dim).to(self.device)
        self.model = nn.Sequential(self.autoencoder, self.clustering_layer)

        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.lr = lr

        # Batch size can't be larger than dataset size: --->
        self.batch_size = min(batch_size, input_dim * 5)
        self.verbose = verbose

    def pretrain_autoencoder(self, X):
        """
        1. Trains the autoencoder using only reconstruction loss (MSE).
        2. Initializes cluster centroids using KMeans on the resulting latent space.
        """
        if self.verbose:
            print("... Pre-training Autoencoder")

        dataset = TensorDataset(torch.from_numpy(X).float())
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        optimizer = Adam(self.autoencoder.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.pretrain_epochs):
            total_loss = 0

            for batch_data in dataloader:
                x = batch_data[0].to(self.device)
                x_hat, _ = self.autoencoder(x)
                loss = loss_fn(x_hat, x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.pretrain_epochs}], Loss: {avg_loss:.4f}")

        # Initialize cluster centroids with KMeans: --->
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).float().to(self.device)
            latent_z = self.autoencoder.encoder(x_tensor).cpu().numpy()

        kmeans = KMeans(n_clusters = self.clustering_layer.n_clusters, n_init = 'auto', random_state = 42)
        kmeans.fit(latent_z)

        self.clustering_layer.centroids.data = torch.from_numpy(kmeans.cluster_centers_).to(self.device)

        if self.verbose:
            print("... Autoencoder pre-trained and centroids initialized.")

    def train(self, X):
        """
        Jointly trains the autoencoder and clustering layer using a
        combined reconstruction and clustering (KL-divergence) loss.
        """
        if self.verbose:
            print("... Jointly training Autoencoder and Clustering Layer")

        dataset = TensorDataset(torch.from_numpy(X).float())
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)
        optimizer = Adam(self.model.parameters(), lr = self.lr)

        reconstruction_loss_fn = nn.MSELoss()
        clustering_loss_fn = nn.KLDivLoss(reduction = 'sum')

        for epoch in range(self.train_epochs):
            total_loss = 0

            for i, batch_data in enumerate(dataloader):
                x = batch_data[0].to(self.device)

                x_hat, z = self.autoencoder(x)
                q = self.clustering_layer(z)

                with torch.no_grad():
                    p = target_distribution(q)

                reconstruction_loss = reconstruction_loss_fn(x_hat, x)
                clustering_loss = clustering_loss_fn(q.log(), p)

                # The paper suggests a weight for the clustering loss. 0.1 is a common value: --->
                loss = 0.1 * clustering_loss + reconstruction_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.train_epochs}], Loss: {avg_loss:.4f}")

        if self.verbose:
            print("... Joint training complete.")

    def predict(self, X):
        """
        Predicts cluster assignments for the input data using the trained model.
        """
        with torch.no_grad():
            x_tensor = torch.from_numpy(X).float().to(self.device)
            _, z = self.autoencoder(x_tensor)
            q = self.clustering_layer(z)
            labels = torch.argmax(q, dim=1).cpu().numpy()

        return labels
