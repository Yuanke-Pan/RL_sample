import torch.nn as nn
import torch.nn.functional as F
import torch
import timm


class RLNet(nn.Module):
    def __init__(self, n_observation, n_actions):
        self.model = timm.create_model("efficientnet_b4",pretrained=True, num_classes=n_actions)
    
    def forward(self, x):
        x = self.model(x)
        return x

class simpleCnn(nn.Module):
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        n_action = 6
    ) -> None:
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper、
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.rand([1, observation_space.shape[0], observation_space.shape[1], observation_space.shape[2]]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU(), nn.Linear(features_dim, n_action))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))