import torch
from generator import Generator

class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, env):
        self.ori_model = model
        self.own_model = Generator(env)
        self.update_params()
        self.own_model.to(env.device)
        self.env = env

    def get_reward(self, trajs, discriminator, rand_var, start_time, start_node_dist):
        """
        Inputs: x, discriminator
            - x: (batch_size, seq_len, 2) input data
            - discriminator: discriminator model
        """
        #with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        size = trajs.size()
        seq_len = size[1]
        batch_size = size[0]
        rewards = torch.zeros(batch_size, seq_len, device=self.env.device)
        for i in range(self.env.rollout_num):
            for l in range(1, seq_len):
                prefix = trajs[:, 0:l]
                samples, _, = self.own_model.sample(seq_len, prefix, rand_var, start_time, start_node_dist)
                pred = discriminator(samples)
                rewards[:, l-1] += pred[:, 1]
            # for the last token
            #print(i)
            pred = discriminator(trajs.detach())
            rewards[:, seq_len-1] += pred[:, 1]
                
        rewards /= (1.0 * self.env.rollout_num) # batch_size * seq_len
        #rewards = torch.transpose((rewards), 0, 1) / (1.0 * self.env.rollout_num) # batch_size * seq_len
        #if self.env.rescale_rewards:
            #rewards = self.rescale_rewards(rewards)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        return rewards

    # takes too much time
    def rescale_rewards(self, rewards):
        batch_size = rewards.size(0)
        traj_len = rewards.size(1)
        for step in range(traj_len):
            step_reward = rewards[:, step]
            _, indices = torch.sort(step_reward, descending=True)
            temp_value = torch.linspace(1, batch_size, batch_size, device=self.env.device)
            temp_value = torch.sigmoid(self.env.sigma * (0.5 - (temp_value / batch_size)))
            step_reward[indices] = temp_value
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            param.data = dic[name]