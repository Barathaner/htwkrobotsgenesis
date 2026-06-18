"""AMP discriminator for style reward (Peng et al. 2021, "AMP: Adversarial Motion Priors")."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        amp_obs_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        lr: float = 1e-4,
        grad_penalty_weight: float = 10.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.amp_obs_dim = amp_obs_dim
        self.grad_penalty_weight = grad_penalty_weight

        layers: list[nn.Module] = []
        in_dim = 2 * amp_obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, s_next], dim=-1))

    def reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """AMP reward ∈[0,1]: max(0, 1 - 0.25*(D-1)^2). High when transition looks like reference."""
        with torch.no_grad():
            d = self.forward(s, s_next)
            return torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0).squeeze(-1)

    def update(
        self,
        real_s: torch.Tensor,
        real_sn: torch.Tensor,
        fake_s: torch.Tensor,
        fake_sn: torch.Tensor,
    ) -> dict[str, float]:
        real_logit = self.forward(real_s, real_sn)
        fake_logit = self.forward(fake_s, fake_sn)

        # Least-squares GAN (LSGAN): pushes D→1 for real, D→0 for fake.
        # Matches the reward formula max(0, 1-0.25*(D-1)^2) from Peng et al. 2021 AMP.
        # BCE would push logits to ±∞, making the reward always 0 after training.
        loss_real = F.mse_loss(real_logit, torch.ones_like(real_logit))
        loss_fake = F.mse_loss(fake_logit, torch.zeros_like(fake_logit))

        # Gradient penalty on real samples (keeps discriminator Lipschitz)
        inp = torch.cat([real_s, real_sn], dim=-1).detach().requires_grad_(True)
        grad = torch.autograd.grad(self.net(inp).sum(), inp, create_graph=True)[0]
        gp = grad.pow(2).sum(dim=-1).mean()

        loss = loss_real + loss_fake + self.grad_penalty_weight * gp
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "amp/disc_loss": loss.item(),
            "amp/disc_real": real_logit.mean().item(),   # should converge toward 1.0
            "amp/disc_fake": fake_logit.mean().item(),   # should converge toward 0.0
        }
