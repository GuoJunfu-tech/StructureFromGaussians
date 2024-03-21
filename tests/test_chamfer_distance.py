import torch
from torch.cuda.amp import autocast


def batched_chamfer_distance(set1, set2, batch_size=512):
    """
    Compute Chamfer distance between two sets of points in PyTorch, supporting gradient tracking.

    Args:
    - set1: Tensor of shape (N, D) representing the first set of points.
    - set2: Tensor of shape (M, D) representing the second set of points.
    - batch_size: Size of the batch for distance computations.

    Returns:
    - The Chamfer distance.
    """
    N, D = set1.shape
    M, _ = set2.shape
    device = set1.device

    # Initialize tensors for minimum distances
    min_dists_1_to_2 = torch.full((N,), float("inf"), device=device)
    min_dists_2_to_1 = torch.full((M,), float("inf"), device=device)

    # Compute in batches to save memory
    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        subset_1 = set1[start_idx:end_idx]

        for start_jdx in range(0, M, batch_size):
            end_jdx = start_jdx + batch_size
            subset_2 = set2[start_jdx:end_jdx]

            # Compute pairwise distance
            dists = torch.cdist(subset_1, subset_2)

            # Update minimum distances
            min_dists_1_to_2[start_idx:end_idx] = torch.min(
                torch.cat(
                    (min_dists_1_to_2[start_idx:end_idx].unsqueeze(1), dists), dim=1
                ),
                dim=1,
            ).values
            min_dists_2_to_1[start_jdx:end_jdx] = torch.min(
                torch.cat(
                    (min_dists_2_to_1[start_jdx:end_jdx].unsqueeze(1), dists.t()), dim=1
                ),
                dim=1,
            ).values

    # Compute the Chamfer distance
    chamfer_dist = torch.mean(min_dists_1_to_2) + torch.mean(min_dists_2_to_1)

    return chamfer_dist


# Example usage with gradient tracking
set1 = torch.randn(100000, 3, requires_grad=True, device="cuda")
set2 = torch.randn(20000, 3, requires_grad=True, device="cuda")

import time

start_time = time.time()
chamfer_distance = batched_chamfer_distance(set1, set2)
cd1 = time.time()
print(f"Chamfer Distance: {chamfer_distance.item()} with {cd1-start_time} secs")
start_time = time.time()
with autocast():
    chamfer_distance = batched_chamfer_distance(set1, set2)
    cd1 = time.time()
    print(
        f"Chamfer Distance cuda: {chamfer_distance.item()} with {cd1-start_time} secs"
    )


# Example backward pass
chamfer_distance.backward()
