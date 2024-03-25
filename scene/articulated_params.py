import torch
import math


class Revolute:
    def __init__(self) -> None:
        # Initial
        self.axis = torch.tensor(
            [0, 1, 0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        self.pivot = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=torch.float32,
            requires_grad=True,
            device="cuda",
        )
        # self.theta = torch.tensor(
        #     [math.pi / 2],
        #     dtype=torch.float32,
        #     requires_grad=True,
        #     device="cuda",
        # )
        self.optimizer = torch.optim.Adam([self.axis, self.pivot], lr=0.005, eps=2e-15)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9
        )
