import random
import math
import torch

from utils.rotation_utils import RotationOperator


def test_rotation_operator():
    # theta = random.rand() * math.pi * 2

    rotate = RotationOperator()
    for test_id in range(10):
        print(f"[testRotationOperator]::Testing with {test_id} intervals")
        # setup
        bn = random.randint(1, 100000)  # batch size
        axis = torch.rand(3)
        axis = axis / torch.norm(axis)
        pivot = torch.rand(3)
        xyz = torch.rand(bn, 3)
        q = torch.rand(bn, 4)
        q = q / torch.norm(q, dim=1, keepdim=True)

        intervals = random.randint(2, 10)

        rotate_interval = 2 * math.pi / intervals
        theta = torch.tensor(rotate_interval, dtype=torch.float32)
        q_origin = q.clone()
        xyz_origin = xyz.clone()
        for i in range(intervals):
            new_xyz = rotate.get_new_location(xyz, axis, pivot, theta)

            q = rotate.get_new_quaternion(q, axis, theta)

        # test
        diff_q_p = (q_origin - q).sum().item()
        diff_q_n = (q_origin + q).sum().item()
        diff_xyz = (xyz_origin - new_xyz).sum().item()

        assert diff_q_p <= 1e-2 or diff_q_n <= 1e-2
        assert diff_xyz <= 1e-2


if __name__ == "__main__":
    # Assuming q_rot is a 4-element tensor representing a quaternion
    q_rot = torch.randn(4)  # Example quaternion

    # Assuming q is a Nx4 tensor representing a batch of quaternions
    N = 5
    q = torch.randn(N, 4)  # Batch of quaternions

    # Correctly expand q_rot to match the batch size of q
    print(q.size())
    q_rot_expanded = q_rot.unsqueeze(0).expand(q.size(0), -1)

    print(q_rot_expanded.shape)  # Should print torch.Size([N, 4])
