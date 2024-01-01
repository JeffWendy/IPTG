import random
import torch

class Dataset:
    def __init__(self, data_dict, num_batches, lens_dist, SOT, EOT, seed_generator, device, batch_size=32):
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.num_batches = num_batches
        self.idx = 0
        self.data_num = 0
        self.lens_dist = lens_dist
        self.SOT = SOT
        self.EOT = EOT
        self.seed_generator = seed_generator
        self.device = device
        self.reset()
        pass

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        self.data_num = 0

    def multinomial(self, dist, num):
        return torch.multinomial(dist, num, replacement=True, generator=self.seed_generator).squeeze()

    def next(self):
        if self.idx >= self.num_batches:
            raise StopIteration
        lens_dist = self.lens_dist
        seq_len = self.multinomial(lens_dist, 1).item()
        while not seq_len in self.data_dict:
            seq_len = self.multinomial(lens_dist, 1).item()
        random.shuffle(self.data_dict[seq_len])
        traj_num = self.batch_size
        if traj_num > len(self.data_dict[seq_len]):
            traj_num = len(self.data_dict[seq_len])
        d = self.data_dict[seq_len][0 : self.batch_size]
        d = torch.tensor(d, dtype=torch.long, device=self.device)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.ones(traj_num, 1, dtype=torch.int64, device=self.device) * self.SOT, d], dim=1)
        target = torch.cat([d, torch.ones(traj_num, 1, dtype=torch.int64, device=self.device) * self.EOT], dim=1)

        self.idx += 1
        self.data_num += traj_num
        return data, target

class DatasetFactory:
    """ Toy data iter to load digits """

    def __init__(self, data_file, labels, arg):
        super(DatasetFactory, self).__init__()

        self.whole_data_dict, self.traj_max_len = self._read_file(data_file) # {seq_len: [[]]}
        self.num_batches = arg.num_batches
        self.idx = 0
        self.data_num = 0
        self.lens_dist = arg.lens_dist
        self.vocab_size = arg.vocab_size
        self.SOT = arg.SOT
        self.EOT = arg.EOT
        self.labels = labels
        self.arg = arg

    def _read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        dct = {}
        max_length = 0
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            if len(l) == 1:
                l = int(l[0])
                seq_len = l
                max_length = max(max_length, seq_len)
                if not seq_len in dct:
                    dct[seq_len] = []
                continue
            dct[seq_len].append(l)
        return dct, max_length
    
    def set_labels(self, labels):
        self.labels = labels

    def build_dataset_for_cluster(self, cluster_idx):
        trajs = self.whole_data_dict
        subset_trajs = {}
        max_length = self.traj_max_len
        traj_idx = -1
        labels = self.labels

        len_dist = torch.zeros(max_length + 1).to(self.arg.device)
        node_dist = torch.zeros(self.vocab_size).to(self.arg.device)

        for l in range(1, max_length + 1):
            if not (l in trajs):
                continue
            for traj in trajs[l]:
                traj_idx += 1
                if cluster_idx != -1 and labels[traj_idx] != cluster_idx:
                    continue
                if not (l in subset_trajs):
                    subset_trajs[l] = []
                if cluster_idx != -1:
                    subset_trajs[l].append(traj)
                len_dist[l] += 1
                for step in traj:
                    if step in [0, 16, 24, 47, 23, 15, 9, 65]:
                        continue
                    node_dist[step] += 1

        if cluster_idx == -1:
            subset_trajs = trajs
            
        return Dataset(subset_trajs, self.num_batches, len_dist, 
                       self.SOT, self.EOT, self.arg.seed_generator, 
                       self.arg.device, self.arg.batch_size), \
                        node_dist, len_dist
    
# 先读取文件build_dataset
# todo: dataloder， evaluation，topoloss, disloss