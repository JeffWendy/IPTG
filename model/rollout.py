import copy
import numpy as np
import torch


class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.own_model.rollout = True

    def get_reward(self, x, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = torch.cat((torch.ones(batch_size, 1, dtype=torch.long, device=torch.device('cuda:0')) * 2751, x[:, 0:l]), dim=1)
                samples, _, _ = self.own_model.sample(batch_size, seq_len, data)
                pred = discriminator(samples)
                pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]