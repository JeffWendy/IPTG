import math
import random
import torch


class GenDataIter:
    """ Toy data iter to load digits """

    def __init__(self, data_file, batch_size, arg):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_dict = self.read_file(data_file) # {seq_len: [[]]}
        self.num_batch = arg.num_batch
        self.idx = 0
        self.data_num = 0
        self.lens_dist = arg.lens_dist
        self.num_nodes = arg.num_nodes
        self.arg = arg
        self.reset()

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        self.data_num = 0

    def multinomial(self, dist, num):
        return torch.multinomial(dist, num, replacement=True, generator=self.arg.seed_generator).squeeze()

    def next(self):
        if self.idx >= self.num_batch:
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
        d = torch.tensor(d, dtype=torch.long, device=self.arg.device)

        # 0 is prepended to d as start symbol
        data = torch.cat([torch.ones(traj_num, 1, dtype=torch.int64, device=self.arg.device) * self.num_nodes, d], dim=1)
        target = torch.cat([d, torch.ones(traj_num, 1, dtype=torch.int64, device=self.arg.device) * self.num_nodes + 1], dim=1)

        self.idx += 1
        self.data_num += traj_num
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        dct = {}
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            if len(l) == 1:
                seq_len = int(l[0])
                if not seq_len in dct:
                    dct[seq_len] = []
                continue
            dct[seq_len].append(l)
        return dct

class DisDataIter:
    """ Toy data iter to load digits """

    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] +\
                        [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = math.ceil(self.data_num / self.batch_size)
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
    
    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.tensor(data)
        label = torch.tensor(label)
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l) 
        return lis
