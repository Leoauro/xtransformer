from torch import nn


class BasicExpert(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int):
        super(BasicExpert, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(self.in_features, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(self.hidden_dim, self.out_features)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        out = self.relu2(out)
        return out
