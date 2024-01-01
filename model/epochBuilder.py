import sys
import os.path
from numpy import dtype
import torch
from generator import Generator

sys.path.append(os.path.split(sys.path[0])[0])

# 用于训练判别器、测试判别器
class EpochBuilder():
    def __init__(self, env, generator:Generator, train_dataset):
        self.env = env
        self.batch_size = env.batch_size
        self.dataset = train_dataset
        self.generator = generator
        self.num_batch = env.num_batch

    def __iter__(self):
        # positive/negatives数据为数列，每个元素为同一长度的半个batch的轨迹（half_batvh * traj_len * 2）
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count >= self.num_batch:
            raise StopIteration()
        self.batch_count += 1
        return self._generate_batch()

    def _shuffle(self, cat_neg_pos, label, batch_size):
        idx = torch.randint(0, batch_size-1, (batch_size,))
        cat_neg_pos = cat_neg_pos[idx]
        label = label[idx]

    def _generate_batch(self):
        # 先查找dataset，看len长的轨迹有多少个，如果少，就取所有，并把traj_num设为dataset len 轨迹数
        # 如果多，就设为nums_trajs_per_batch
        # 再采样起始时间和地点，生成同样数量的neg，和pos组成一个批次，作为一个batch，添加到epoch builder的数列里面，同时添加label
        joint_dist = self.env.startTime_trajLens_joint_dist
        lens_dist = self.env.lens_dist
        entrance_distribution =  self.env.entrance_distribution
        start_nodes = []

        seq_len = self.multinomial(lens_dist, 1).item()
        start_time_dist = joint_dist[:, seq_len]

        traj_num = self.batch_size // 2 
        dataset_traj_num = self.dataset[seq_len].size(0)
        if (torch.sum(start_time_dist) < traj_num):
            traj_num = torch.sum(start_time_dist).to(torch.long).item()
        if (dataset_traj_num < traj_num):
            traj_num = dataset_traj_num

        start_time = self.multinomial(start_time_dist, traj_num)
        for i in range(traj_num):
            entrance = self.multinomial(entrance_distribution[:, start_time[i]], 1).squeeze()
            entrance = self.env.entrance_idx[entrance]
            start_nodes.append(entrance.unsqueeze(dim=0))

        start_nodes = torch.stack(start_nodes, dim=0)
        start_node_dist = torch.zeros(traj_num, self.env.num_nodes, dtype=torch.float, device=self.env.device)

        rand_var = torch.rand(traj_num, self.env.gen_random_size, device=self.env.device)
        neg_samples, _, = self.generator.sample(seq_len, start_nodes, rand_var, start_time, start_node_dist)
        pos_dist = torch.ones(dataset_traj_num, device=self.env.device)
        pos_idx = torch.multinomial(pos_dist, traj_num, generator=self.env.seed_generator).squeeze()
        pos_samples = self.dataset[seq_len][pos_idx, :]
        tmp = pos_samples.to('cpu').numpy()
        samples = torch.cat((neg_samples, pos_samples), dim=0)
        labels = torch.cat((torch.zeros(traj_num, device=self.env.device, dtype=torch.long), torch.ones(traj_num, device=self.env.device, dtype=torch.long)))
        self._shuffle(samples, labels, 2 * traj_num)
        
        return samples, labels
    
    def multinomial(self, dist, num):
        return torch.multinomial(dist, num, replacement=True, generator=self.env.seed_generator).squeeze()