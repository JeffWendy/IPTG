import torch.nn as nn

class TOPOLoss(nn.Module):
    """
    Topological loss
    """

    def __init__(self):
        super(TOPOLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, topo_errs, target):
        """
        Inputs:
            - pred: (batch_size, seq_len), 
            - target : (batch_size, seq_len), 
            - reward : (batch_size, ), reward of each whole sentence
        Output: 
            - loss: (1,)
        """
        node_count = topo_errs.size(2)
        loss = self.mseloss(topo_errs.view(-1, node_count), target.view(-1, node_count))
        return loss