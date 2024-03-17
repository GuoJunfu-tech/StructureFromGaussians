import torch
import math


class Revolute:
    def __init__(self) -> None:
        # Initial
        self.axis = torch.tensor([0, 0, 1], dtype=torch.float32, requires_grad=True)
        self.pivot = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        self.theta = torch.tensor(
            [math.pi / 2], dtype=torch.float32, requires_grad=True
        )
        self.optimizer = torch.optim.Adam(
            [self.axis, self.theta, self.pivot], lr=0.0005
        )
