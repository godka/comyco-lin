import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_NUM = 128
ACTION_EPS = 1e-4


class _ActorBackbone(nn.Module):
    """Neural network that mirrors the original TensorFlow architecture."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        s_info, s_len = state_dim

        if s_info != 6:
            raise ValueError(f"Expected state info dimension 6, got {s_info}")

        self.action_dim = action_dim

        # Fully connected branches for scalar features.
        self.split_0 = nn.Linear(1, FEATURE_NUM)
        self.split_1 = nn.Linear(1, FEATURE_NUM)
        self.split_5 = nn.Linear(1, FEATURE_NUM)

        # Temporal branches for historical signals.
        self.split_2 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)
        self.split_3 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)
        self.split_4 = nn.Conv1d(1, FEATURE_NUM, kernel_size=1)

        # Merge layers.
        merged_dim = FEATURE_NUM * 5  # three conv branches + two fc branches
        merged_dim += FEATURE_NUM  # split_5 branch
        self.merge_fc = nn.Linear(merged_dim, FEATURE_NUM)
        self.out_fc = nn.Linear(FEATURE_NUM, action_dim)

    def forward(self, x):
        # x shape: (batch, 6, 8)
        split_0_in = x[:, 0, -1].unsqueeze(-1)
        split_1_in = x[:, 1, -1].unsqueeze(-1)
        split_5_in = x[:, 5, -1].unsqueeze(-1)

        split_2_in = x[:, 2:3, :]
        split_3_in = x[:, 3:4, :]
        split_4_in = x[:, 4:5, :self.action_dim]

        split_0 = F.relu(self.split_0(split_0_in))
        split_1 = F.relu(self.split_1(split_1_in))
        split_5 = F.relu(self.split_5(split_5_in))

        split_2 = F.relu(self.split_2(split_2_in))
        split_3 = F.relu(self.split_3(split_3_in))
        split_4 = F.relu(self.split_4(split_4_in))

        split_2 = torch.flatten(split_2, start_dim=1)
        split_3 = torch.flatten(split_3, start_dim=1)
        split_4 = torch.flatten(split_4, start_dim=1)

        merged = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], dim=1)
        merged = F.relu(self.merge_fc(merged))
        logits = self.out_fc(merged)
        probs = F.softmax(logits, dim=1)
        return probs


class Network:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        self.model = _ActorBackbone(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _prepare_state(self, state):
        tensor = torch.from_numpy(state).float().to(self.device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            tensor = self._prepare_state(state)
            probs = self.model(tensor)
        return probs.cpu().numpy()[0]

    def train_step(self, s_batch, a_batch):
        self.model.train()

        states = self._prepare_state(s_batch)
        actions = torch.from_numpy(a_batch).float().to(self.device)

        probs = self.model(states)
        probs = torch.clamp(probs, ACTION_EPS, 1.0 - ACTION_EPS)
        log_probs = torch.log(probs)

        cross_entropy = -(actions * log_probs).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = cross_entropy + 0.2 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), entropy.item()

    def save_model(self, path):
        torch.save({"model_state_dict": self.model.state_dict()}, path)
        return path

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)

