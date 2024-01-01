import torch
import torch.nn as nn


class PGLoss(nn.Module):
    """
    Pseudo-loss that gives corresponding policy gradients (on calling .backward()) 
    for adversial training of Generator
    """

    def __init__(self, env):
        super(PGLoss, self).__init__()
        self.env = env

    def forward(self, log_pred, target, reward, seq_len):
        """
        Inputs:
            - log_pred: (batch_size * seq_len, node_count), 
            - target: (batch_size, seq_len), 
            - reward: (batch_size, seq_len), reward of each whole sentence,
            - seq_len: (1, ), sequence number
        Output:
            - loss: (1, )
        """
        one_hot = torch.zeros(log_pred.size(), dtype=torch.bool, device=self.env.device)
        one_hot.scatter_(1, target.data.view(-1, 1), 1)
        target_pred = torch.masked_select(log_pred, one_hot)
        loss = target_pred * reward.contiguous().view(-1)
        loss = -torch.sum(loss)/(seq_len * target.size(0))
        return loss