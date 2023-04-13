import torch
from torch import Tensor
import torch.nn as nn

class DenseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(DenseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # flatten_tensor = self.flatten(input_tensor)
        fc1_out = self.fc1(input_tensor)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        return fc3_out

