import torch
import torch.nn.functional as F
from torch import Tensor


class SharedField:
    def __init__(self, dim: int, rho: float = 0.1, eta: float = 0.5):
        self.dim = dim
        self.rho = rho  # evaporation rate
        self.eta = eta  # deposit weight
        self.state = torch.zeros(dim)

    def is_zero(self) -> bool:
        return self.state.norm().item() < 1e-8

    def read(self) -> Tensor:
        return self.state.clone()

    def update(self, deposited: list[Tensor]) -> None:
        self.state = (1 - self.rho) * self.state  # unconditional evaporation

        if deposited:
            mean_deposit = torch.stack(deposited).mean(dim=0)
            self.state = self.state + self.eta * mean_deposit

        norm = self.state.norm()
        if norm > 0:
            self.state = self.state / norm
