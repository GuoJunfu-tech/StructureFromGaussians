import numpy as np
import torch
import math
from typing import Union


class RotationOperator:

    @staticmethod
    def get_rotation_matrix(axis: Union[torch.Tensor, np.ndarray], theta):
        if isinstance(axis, np.ndarray):
            axis = axis / np.linalg.norm(axis)  # normalize
            kx, ky, kz = axis[0], axis[1], axis[2]
            cos, sin = np.cos(theta), np.sin(theta)
            R = np.zeros((3, 3))
            R[0, 0] = cos + (kx**2) * (1 - cos)
            R[0, 1] = kx * ky * (1 - cos) - kz * sin
            R[0, 2] = kx * kz * (1 - cos) + ky * sin
            R[1, 0] = kx * ky * (1 - cos) + kz * sin
            R[1, 1] = cos + (ky**2) * (1 - cos)
            R[1, 2] = ky * kz * (1 - cos) - kx * sin
            R[2, 0] = kx * kz * (1 - cos) - ky * sin
            R[2, 1] = ky * kz * (1 - cos) + kx * sin
            R[2, 2] = cos + (kz**2) * (1 - cos)
        elif isinstance(axis, torch.Tensor):
            axis = axis / torch.linalg.norm(axis)  # normalize
            kx, ky, kz = axis[0], axis[1], axis[2]
            cos, sin = torch.cos(theta), torch.sin(theta)
            one = torch.tensor(1.0, dtype=axis.dtype)
            R = torch.zeros((3, 3))
            R[0, 0] = cos + (kx**2) * (one - cos)
            R[0, 1] = kx * ky * (one - cos) - kz * sin
            R[0, 2] = kx * kz * (one - cos) + ky * sin
            R[1, 0] = kx * ky * (one - cos) + kz * sin
            R[1, 1] = cos + (ky**2) * (one - cos)
            R[1, 2] = ky * kz * (one - cos) - kx * sin
            R[2, 0] = kx * kz * (one - cos) - ky * sin
            R[2, 1] = ky * kz * (one - cos) + kx * sin
            R[2, 2] = cos + (kz**2) * (one - cos)
        else:
            raise ValueError(
                f"axis and theta should be numpy or torch matrix, now it is {type(axis)}"
            )

        return R

    def get_new_location(self, x, axis, pivot_point, theta):
        """
        Computes the new location of a point after rotation in a revolute joint.

        Args:
        x: Point coordinates as a 3D vector (numpy array).
        u: Revolute axis vector as a unit vector (numpy array).
        q: Pivot point coordinates as a 3D vector (numpy array).
        theta: Rotation angle about the revolute axis.

        Returns:
        A numpy array representing the new location of the point.
        """
        # x = np.asarray(x)
        if isinstance(x, np.ndarray):
            p_local = np.transpose(x - pivot_point)
            R = self.get_rotation_matrix(axis, theta)
            p_local_rotated = R.dot(p_local)
            x_new = np.transpose(p_local_rotated) + pivot_point
        elif isinstance(x, torch.Tensor):
            p_local = torch.transpose(x - pivot_point, 0, 1)
            R = self.get_rotation_matrix(axis, theta)
            # p_local_rotated = R.dot(p_local)
            p_local_rotated = torch.mm(R, p_local)
            x_new = torch.transpose(p_local_rotated, 0, 1) + pivot_point
        else:
            raise ValueError("x must be a numpy matrix or torch matrix!")

        return x_new

    def get_new_quaternion(
        self,
        q: Union[torch.Tensor, np.ndarray],
        axis: Union[torch.Tensor, np.ndarray],
        theta: Union[torch.Tensor, np.ndarray, float],
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(axis, np.ndarray):
            q_rot = np.array(
                [
                    math.cos(theta / 2),
                    axis[0] * math.sin(theta / 2),
                    axis[1] * math.sin(theta / 2),
                    axis[2] * math.sin(theta / 2),
                ]
            )
        elif isinstance(axis, torch.Tensor):
            q_rot = torch.tensor(
                [
                    math.cos(theta / 2),
                    axis[0] * torch.sin(theta / 2),
                    axis[1] * torch.sin(theta / 2),
                    axis[2] * torch.sin(theta / 2),
                ]
            )
        else:
            raise ValueError(
                f"Axis must be np or torch matrix, now it is {type(axis)}."
            )

        return q_rot * q


if __name__ == "__main__":
    axis = np.array([0, 0, 1])
    pivot = np.array([0, 0, 0])
    theta = math.pi / 2

    print(RotationOperator.get_rotation_matrix(axis, theta))
