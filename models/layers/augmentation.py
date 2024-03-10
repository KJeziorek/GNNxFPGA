import torch

class RandomHorizontalFlip(torch.nn.Module):
    """Applies a random horizontal flip to the nodes with a probability of `p`."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, nodes, features, dim):
        if torch.rand(1) < self.p:
            # Flip the nodes horizontally and adjust positions
            nodes[:, 0] *= -1
            nodes[:, 0] += dim
            nodes[:, 0] -= nodes[:, 0].min()
        return nodes, features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomPolarityFlip(torch.nn.Module):
    """Flips the polarity of features with a probability of `p`."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, nodes, features, dim):
        if torch.rand(1) < self.p:
            features *= -1
        return nodes, features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomRotationEvent(torch.nn.Module):
    """Rotates the events randomly within a specified angle range."""
    def __init__(self, angle=5):
        super().__init__()
        self.angle = angle

    def forward(self, nodes, features, dim):
        angle = (torch.rand(1) * 2 - 1) * self.angle
        angle_rad = angle * torch.pi / 180  # Convert angle to radians

        x, y = nodes[:, 0], nodes[:, 1]

        # Apply rotation
        nodes[:, 0] = x * torch.cos(angle_rad).to(nodes.device) - y * torch.sin(angle_rad).to(nodes.device)
        nodes[:, 1] = x * torch.sin(angle_rad).to(nodes.device) + y * torch.cos(angle_rad).to(nodes.device)

        # Filter out events outside the sensor dimensions
        mask = (nodes[:, 0] > 0) & (nodes[:, 0] < dim) & (nodes[:, 1] > 0) & (nodes[:, 1] < dim)
        nodes, features = nodes[mask], features[mask]
        return nodes, features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle={self.angle})"