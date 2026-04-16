import torch
import torch.nn.functional as F
from torch import Tensor


class Agent:
    def __init__(self, agent_id: int, dim: int,
                 beta: float = 0.05,
                 alpha: float = 0.1,
                 theta: float = 0.3):
        self.id = agent_id
        self.dim = dim
        self.beta = beta    # observation weight
        self.alpha = alpha  # field reading weight
        self.theta = theta  # gating threshold (cosine distance)
        self.belief = torch.zeros(dim)

    def _cosine_distance(self, a: Tensor, b: Tensor) -> float:
        return 1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    def step(self, obs: Tensor, field: Tensor, field_is_zero: bool) -> Tensor | None:
        """
        One round update.
        Returns deposited belief (Tensor) if agent deposits, else None.
        """
        # (1) absorb local observation
        self.belief = (1 - self.beta) * self.belief + self.beta * obs

        # (2) deposit decision (before reading field)
        if field_is_zero or self._cosine_distance(self.belief, field) > self.theta:
            deposit = self.belief.clone()
        else:
            deposit = None

        # (3) read from field
        if not field_is_zero:
            self.belief = (1 - self.alpha) * self.belief + self.alpha * field

        # (4) normalize
        norm = self.belief.norm()
        if norm > 0:
            self.belief = self.belief / norm

        return deposit
