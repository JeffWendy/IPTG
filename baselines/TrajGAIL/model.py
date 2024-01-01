import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# batch_first:
class StateSeqEmb(nn.Module):
    def __init__(self, state_dim, action_dim, hidden, num_layers):
        super(StateSeqEmb, self).__init__()
        self.state_dim = state_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.action_dim = action_dim

        self.state_emb = nn.Embedding(self.state_dim + 1, self.hidden, padding_idx=self.state_dim)
        self.rnncell = nn.GRU(self.hidden, self.hidden, self.num_layers, batch_first=True)

        self.activation = torch.relu

    def forward(self, state_seq, seq_len):
        x_emb = self.state_emb(state_seq)
        packed_input = pack_padded_sequence(
            x_emb, seq_len.tolist(), batch_first=True)
        h, x_rnn = self.rnncell(packed_input)
        return h, x_rnn

class Policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden: int, origins, start_code, 
            env, disttype="categorical", num_layers=3):
        super(Policy_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.origins = origins
        self.origin_dim = origins.shape[0]
        self.start_code = start_code
        self.env = env

        self.fc1 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc3 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.activation = torch.relu

        self.fc4 = nn.Linear(in_features=self.hidden, out_features=self.action_dim)
        self.prob_dim = self.action_dim

        self.action_domain = torch.zeros(len(self.env.states), self.env.max_actions).long()
        for i in range(len(self.env.states)):
            s0 = self.env.states[i]
            if not s0 == self.env.terminal:
                self.action_domain[i, list(self.env.netconfig[s0].keys())] = 1
        self.action_domain = torch.nn.Parameter(self.action_domain, requires_grad=False)

        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)
        self.env = env

    def forward(self, state_seq, seq_len):
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        x = self.activation(self.fc1(x_rnn))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))

        last_states = state_seq[torch.arange(state_seq.size(0)), seq_len-1]
        # masked_logit是当前 
        masked_logit = self.fc4(x).masked_fill(
            (1-self.action_domain[last_states]).bool(), -1e32)
        prob = torch.nn.functional.softmax(masked_logit, dim=1)

        action_dist = torch.distributions.Categorical(prob)

        return action_dist

    def pretrain_forward(self, state_seq, seq_len):
        h, _ = self.StateSeqEmb(state_seq, seq_len)
        h, seq_len1 = pad_packed_sequence(h, True)

        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.activation(self.fc3(h))
        prob = self.activation(self.fc4(h))

        prob = prob.view((state_seq.size(0), state_seq.size(1), self.prob_dim))

        return prob

class Value_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden: int, num_layers=3):
        super(Value_net, self).__init__()
        """
        :param name: string
        :param env: gym env
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.state_emb = nn.Embedding(self.state_dim, self.hidden)
        self.activation = torch.relu

        self.fc1 = nn.Linear(in_features=self.hidden + self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc3 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc4 = nn.Linear(in_features=self.hidden, out_features=1)
        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(
            self.state_dim, self.action_dim, self.hidden, num_layers)

    def one_hotify(self, longtensor, dim):
        if list(self.parameters())[0].is_cuda:
            one_hot = torch.cuda.FloatTensor(longtensor.size(0), dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0), dim).to()
        one_hot.zero_()
        one_hot.scatter_(1, longtensor.unsqueeze(1).long(), 1)
        return one_hot

    def forward(self, state_seq, action, seq_len):
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)
        x_rnn = x_rnn[self.num_layers-1, :, :]

        action = self.one_hotify(action, self.action_dim)

        x = torch.cat([x_rnn, action], dim=1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        value = self.fc4(x)
        return value

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden:int, disttype = "categorical", num_layers = 3):
        super(Discriminator, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.disttype = disttype

        self.hidden = hidden
        self.disttype = disttype
        self.state_emb = nn.Embedding(self.state_dim, self.hidden)

        self.activation = torch.relu

        self.fc1 = nn.Linear(in_features=self.hidden + self.action_dim, out_features=self.hidden)
        self.fc2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc3 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.fc4 = nn.Linear(in_features=self.hidden, out_features=1)

        self.num_layers = num_layers
        self.StateSeqEmb = StateSeqEmb(self.state_dim, self.action_dim, self.hidden, num_layers)

    def get_reward(self, *args, **kwargs):
        with torch.no_grad():
            if kwargs:
                prob = self.forward(**kwargs)
            elif args:
                prob = self.forward(*args)
            else:
                raise ValueError
            # clamp : 将输入input张量每个元素的夹紧到区间 [min,max]
            return -torch.log(torch.clamp(prob, 1e-10, 1))

    def one_hotify(self, longtensor, dim):
        if list(self.parameters())[0].is_cuda:
            one_hot = torch.cuda.FloatTensor(longtensor.size(0), dim).to()
        else:
            one_hot = torch.FloatTensor(longtensor.size(0), dim).to()
        one_hot.zero_()
        one_hot.scatter_(1, longtensor.long(), 1)
        return one_hot

    def forward(self, state_seq, action, seq_len , act_prob=None):
        _, x_rnn = self.StateSeqEmb(state_seq, seq_len)         
        x_rnn = x_rnn[self.num_layers-1, :, :]

        if act_prob is not None:
            action = act_prob
        else:
            action = self.one_hotify(action.unsqueeze(1), self.action_dim)
        
        # feed the (unfinished or finished) traj and the next action into MLP
        x = torch.cat([x_rnn, action], dim = 1)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        prob = torch.sigmoid(self.fc4(x))
        return prob