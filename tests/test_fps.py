import torch
# from utils.general_utils import farthest_point_sampling


def farthest_point_sampling(tensor, num_samples):
    n = tensor.size(0)
    distances = torch.full((n,), float("inf"), device=tensor.device)
    selected_indices = torch.zeros(num_samples, dtype=torch.long, device=tensor.device)

    # Arbitrarily choose the first point as the starting point
    current_index = torch.randint(0, n, (1,)).item()
    for i in range(num_samples):
        selected_indices[i] = current_index
        current_point = tensor[current_index].unsqueeze(0)
        dist = torch.norm(tensor - current_point, dim=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        current_index = torch.argmax(distances).item()

    return tensor[selected_indices]


def test_fps():
    n = 100000
    tensor = torch.randn(n, 3).to("cuda")

    num_samples = 1000

    print(farthest_point_sampling(tensor, num_samples))


if __name__ == "__main__":
    test_fps()
