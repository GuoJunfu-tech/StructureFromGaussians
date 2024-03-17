import torch
import math


class Revolute:
    def __init__(self) -> None:
        # Initial
        self.axis = torch.tensor(
            [1, 0, 0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.pivot = torch.tensor(
            [0.729, 0.175, -0.15],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.theta = torch.tensor(
            [-math.pi * 160 / 180.0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.optimizer = torch.optim.Adam([self.theta], lr=0.1)
        # self.optimizer = torch.optim.Adam([self.axis, self.theta, self.pivot], lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )
