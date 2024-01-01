import torch
import torch.nn as nn

class DLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward()) 
    for adversial training of Generator
    """

    def __init__(self):
        super(DLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, node_dist, target_dist):
        """
        Inputs: 
            - node_dist: (batch_size, seq_len), 
            - target_dist: (batch_size, seq_len), 
        """
        num_nodes = node_dist.size(2)
        node_dist = node_dist[:, 1:-1, :]
        sum_dist = torch.sum(torch.sum(node_dist, dim=0), dim=0) / (node_dist.size(0) * node_dist.size(1))
        target_dist += 1e-10
        target_dist /= torch.sum(target_dist)
        loss = torch.sum(sum_dist * (torch.log(sum_dist) - torch.log(target_dist)))
        return loss