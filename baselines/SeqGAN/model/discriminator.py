from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Highway architecture based on the pooled feature maps is added. Dropout is adopted.
    """

    def __init__(self, env):
        super(Discriminator, self).__init__()
        self.env = env
        self.node_embed = nn.Embedding(env.num_nodes + 1, env.node_embedding_dim, padding_idx=env.num_nodes)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_f.item(), (f_size.item(), env.node_embedding_dim)) for f_size, num_f in zip(env.filter_sizes, env.num_filters)
        ])
        self.highway = nn.Linear(torch.sum(env.num_filters).item(), torch.sum(env.num_filters).item())
        self.dropout = nn.Dropout(p = env.dis_dropout_prob)
        self.fc = nn.Linear(torch.sum(env.num_filters).item(), 2)

    def forward(self, trajs):
        """
        Inputs: trajs
            - trajs: (batch_size, seq_len)
        Outputs: out
            - out: (batch_size, 2)
        """
        size = trajs.size()
        traj_len = size[1]
        batch_size = size[0]

        if traj_len < 7:
            tmp = torch.ones(batch_size, 7, dtype=torch.long, device=self.env.device)
            tmp[:, :] *= self.env.num_nodes
            tmp[:, 0:traj_len] = trajs
            trajs = tmp

        node_emb = self.node_embed(trajs).unsqueeze(1) # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(node_emb)).squeeze(3) for conv in self.convs] # [batch_size * num_filter * seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)

        highway = self.highway(out)
        transform = torch.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out # sets C = 1 - T
        out = F.softmax(self.fc(self.dropout(out)), dim=1)
        out = torch.log(out)

        return out