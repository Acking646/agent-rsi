"""Optional sparse autoencoder fit at one selected layer/component and anchor."""
import numpy as np

from .backends import Intervention


class SparseAutoencoder:
    def __init__(self, train: np.ndarray, width: int, seed: int, steps: int):
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.mean = torch.from_numpy(train.mean(axis=0).astype("float32"))
        self.scale = max(float(np.sqrt(np.mean((train - self.mean.numpy()) ** 2))), 1e-6)
        data = (torch.from_numpy(train.astype("float32")) - self.mean) / self.scale
        self.encoder = torch.nn.Linear(data.shape[1], width)
        self.decoder = torch.nn.Linear(width, data.shape[1], bias=False)
        opt = torch.optim.Adam([*self.encoder.parameters(), *self.decoder.parameters()], lr=.003)
        for _ in range(steps):
            z = self.encoder(data).relu()
            decoded = self.decoder(z)
            loss = (decoded - data).square().mean() + .01 * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0).clamp_min(1e-8))
        self.encoder.eval()
        self.decoder.eval()
        z = self.encode(train)
        self.stats = {"width": width, "steps": steps, "train_samples": len(train),
                      "train_active_fraction": float((z > 0).mean()),
                      "train_reconstruction_mse": self.mse(train)}

    def encode(self, data: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = (self.torch.from_numpy(data.astype("float32")) - self.mean) / self.scale
            return self.encoder(x).relu().numpy()

    def mse(self, data: np.ndarray) -> float:
        with self.torch.no_grad():
            out = self.decoder(self.torch.from_numpy(self.encode(data))) * self.scale + self.mean
        return float(np.mean((data - out.numpy()) ** 2))

    def patches(self, site, src, ref, train_direction):
        vectors = self.decoder.weight.detach().numpy() * self.scale
        idx = int(np.argmax(np.abs(train_direction @ vectors)))
        source, donor = self.encode(src), self.encode(ref)
        vector = vectors[:, idx]
        # Keep the original reconstruction residual. Edit only one feature's contribution.
        return idx, {
            "sae_feature_patch": Intervention(site, (donor[idx] - source[idx]) * vector, add=True),
            "sae_feature_ablate": Intervention(site, -source[idx] * vector, add=True),
        }

    def save(self, path):
        np.savez_compressed(path, mean=self.mean.numpy(), scale=self.scale,
                            encoder=self.encoder.weight.detach().numpy(),
                            encoder_bias=self.encoder.bias.detach().numpy(),
                            decoder=self.decoder.weight.detach().numpy())
