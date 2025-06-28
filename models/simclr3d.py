import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights


def _conv1(in_c):
    return nn.Conv3d(in_c, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)


class ResNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.model = r3d_18(weights=weights)
        self.model.stem[0] = _conv1(in_channels)
        self.out_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x.flatten(1)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 2048, out_dim: int = 256, pretrained: bool = False):
        super().__init__()
        self.backbone = ResNet3D(in_channels, pretrained=pretrained)
        self.projector = ProjectionMLP(self.backbone.out_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    b = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    rep = torch.cat([z1, z2], 0)
    sim = torch.matmul(rep, rep.t()) / temperature
    lbl = torch.arange(b, device=z1.device)
    lbl = torch.cat([lbl + b, lbl])
    sim = sim - sim.max(1, keepdim=True).values.detach()
    mask = torch.eye(2 * b, device=z1.device, dtype=torch.bool)
    exp_sim = sim.exp().masked_fill(mask, 0)
    log_den = torch.log(exp_sim.sum(1) + 1e-9)
    pos = sim[torch.arange(2 * b, device=z1.device), lbl]
    return -(pos - log_den).mean()


if __name__ == "__main__":
    x1, x2 = (torch.randn(8, 3, 80, 80, 80) for _ in range(2))
    model = SimCLR()
    z1, z2 = model(x1), model(x2)
    print(z1.shape, nt_xent_loss(z1, z2).item())
