import os
import torch
import time
# from tensorboardX import SummaryWriter
import datetime
import numpy as np
from utils import *

# find_state 去除
# 数据：轨迹就是 grid idx sequences; actions需要统计(可能的actions包括八邻域、电梯联通，需要刨除不能联通的actions);


class GAILRNNTrain():
    def __init__(self, env, Policy, Value, Discrim, pad_idx, args):
        """
                :param Policy:
                :param Old_Policy:
                :param gamma:
                :param clip_value:x
                :param c_1: parameter for value difference
                :param c_2: parameter for entropy bonus
                """
        self.env = env
        self.Policy = Policy
        self.Value = Value
        self.Discrim = Discrim

        self.lr = args.learning_rate
        self.eps = args.eps
        self.num_discrim_update = args.num_discrim_update
        self.num_gen_update = args.num_gen_update
        self.gamma = args.gamma
        self.c_1 = args.c_1
        self.c_2 = args.c_2

        self.policy_opt = torch.optim.Adam(
            self.Policy.parameters(), lr=self.lr, eps=self.eps)
        self.value_opt = torch.optim.Adam(
            self.Value.parameters(), lr=self.lr, eps=self.eps)
        self.discrim_opt = torch.optim.Adam(
            self.Discrim.parameters(), lr=self.lr, eps=self.eps)
        self.pretrain_opt = torch.optim.Adam(
            self.Policy.parameters(), lr=self.lr, eps=self.eps)

        self.value_criterion = torch.nn.MSELoss()
        self.discrim_criterion = torch.nn.BCELoss()

        now = datetime.datetime.now()

        self.summary_cnt = 0
        self.rnn_summary_cnt = 0
        self.pad_idx = pad_idx
        self.num_posterior_update = 2
        self.args = args

        self.find_state = lambda x: env.states.index(x) if x != -1 else pad_idx
        self.nettransition = -1 * \
            torch.ones(size=(len(self.env.states),
                       self.env.max_actions)).long()
        for key in self.env.netconfig.keys():
            idx = self.find_state(key)
            for key2 in self.env.netconfig[key]:
                self.nettransition[idx, key2] = self.find_state(
                    self.env.netconfig[key][key2])

    def set_device(self, device):
        self.device = device

    def pretrain(self, exp_trajs, find_state, device):
        """
        Params:
            -find_state:
                def find_state(x): return env.states.index(
                    x) if x != -1 else pad_idx
                find_state = np.vectorize(find_state)
        """
        pretrain_trajs = []
        for episode in exp_trajs:
            pretrain_trajs.append([(x.cur_state, x.action) for x in episode])
        stateseq_len = np.array([len(x) for x in pretrain_trajs])
        try:
            stateseq_maxlen = np.max(stateseq_len)
        except:
            return 0, 0, 0, 0, 0, 0
        stateseq_in = -np.ones((len(pretrain_trajs), stateseq_maxlen))
        stateseq_target = -np.ones((len(pretrain_trajs), stateseq_maxlen))

        for i in range(len(pretrain_trajs)):
            temp = pretrain_trajs[i]
            stateseq_in[i, :len(temp)] = [x[0] for x in temp]
            stateseq_target[i, :len(temp)] = [x[1] for x in temp]

        def get_num_options(x): return len(
            self.env.netconfig[x]) if x != -1 else 0
        get_num_options = np.vectorize(get_num_options)
        num_options = get_num_options(stateseq_in)

        stateseq_in = find_state(stateseq_in)

        stateseq_in = torch.LongTensor(stateseq_in).to(device)
        stateseq_target = torch.LongTensor(stateseq_target).to(device)
        stateseq_len = torch.LongTensor(stateseq_len).to(device)
        num_options = torch.LongTensor(num_options).to(device)

        stateseq_len, sorted_idx = stateseq_len.sort(0, descending=True)
        stateseq_in = stateseq_in[sorted_idx]
        stateseq_target = stateseq_target[sorted_idx]
        num_options = num_options[sorted_idx]

        out = self.unroll_trajectory2(
            num_trajs=len(exp_trajs), max_length=self.args.max_length, batch_size=self.args.batch_size)
        learner_observations = out[0]
        learner_actions = out[1]
        learner_len = out[2]

        learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()))
        learner_act = np.zeros((learner_len.sum()))
        learner_l = np.zeros((learner_len.sum()))
        cnt = 0
        for i0 in range(learner_len.shape[0]):
            for j0 in range(1, learner_len[i0]+1):
                try:
                    learner_obs[cnt, :j0] = learner_observations[i0, :j0]
                    learner_act[cnt] = int(learner_actions[i0][j0-1])
                    learner_l[cnt] = j0
                    cnt += 1
                except:
                    break
        idx = learner_l != 0
        learner_obs = learner_obs[idx]
        learner_act = learner_act[idx]
        learner_l = learner_l[idx]
        learner_obs, learner_act, learner_len = arr_to_tensor(
            find_state, device, learner_obs, learner_act, learner_l)

        exp_obs, exp_act, exp_len = trajs_to_tensor(exp_trajs)
        exp_obs, exp_act, exp_len = arr_to_tensor(
            find_state, device, exp_obs, exp_act, exp_len)

        expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
        learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

        expert_loader = DataLoader(
            dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
        learner_loader = DataLoader(
            dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)

        acc, acc2, loss = self.pretrain_rnn(
            stateseq_in, stateseq_target, stateseq_len, num_options)

        expert_acc, learner_acc, discrim_loss = None, None, None

        for expert_data, learner_data in zip(enumerate(expert_loader), enumerate(learner_loader)):
            sampled_exp_obs, sampled_exp_len, sampled_exp_act = expert_data[1]
            sampled_exp_len, idxs = torch.sort(
                sampled_exp_len, descending=True)
            sampled_exp_obs = sampled_exp_obs[idxs]
            sampled_exp_act = sampled_exp_act[idxs]

            sampled_learner_obs, sampled_learner_len, sampled_learner_act = learner_data[1]
            sampled_learner_len, idxs = torch.sort(
                sampled_learner_len, descending=True)
            sampled_learner_obs = sampled_learner_obs[idxs]
            sampled_learner_act = sampled_learner_act[idxs]

            expert_acc, learner_acc, discrim_loss = \
                self.train_discrim_step(sampled_exp_obs.to(self.device),
                                        sampled_exp_act.to(self.device),
                                        sampled_exp_len.to(self.device),
                                        sampled_learner_obs.to(self.device),
                                        sampled_learner_act.to(self.device),
                                        sampled_learner_len.to(self.device)
                                        )

        return acc, acc2, loss, expert_acc, learner_acc, discrim_loss

    def pretrain_dism(self, exp_trajs, find_state, device, args):
        """
        Params:
            -find_state:
                def find_state(x): return env.states.index(
                    x) if x != -1 else pad_idx
                find_state = np.vectorize(find_state)
        """
        pretrain_trajs = []
        for episode in exp_trajs:
            pretrain_trajs.append([(x.cur_state, x.action) for x in episode])
        stateseq_len = np.array([len(x) for x in pretrain_trajs])
        stateseq_maxlen = np.max(stateseq_len)
        stateseq_in = -np.ones((len(pretrain_trajs), stateseq_maxlen))
        stateseq_target = -np.ones((len(pretrain_trajs), stateseq_maxlen))

        for i in range(len(pretrain_trajs)):
            temp = pretrain_trajs[i]
            stateseq_in[i, :len(temp)] = [x[0] for x in temp]
            stateseq_target[i, :len(temp)] = [x[1] for x in temp]

        def get_num_options(x): return len(
            self.env.netconfig[x]) if x != -1 else 0
        get_num_options = np.vectorize(get_num_options)
        num_options = get_num_options(stateseq_in)

        stateseq_in = find_state(stateseq_in)

        stateseq_in = torch.LongTensor(stateseq_in).to(device)
        stateseq_target = torch.LongTensor(stateseq_target).to(device)
        stateseq_len = torch.LongTensor(stateseq_len).to(device)
        num_options = torch.LongTensor(num_options).to(device)

        stateseq_len, sorted_idx = stateseq_len.sort(0, descending=True)
        stateseq_in = stateseq_in[sorted_idx]
        stateseq_target = stateseq_target[sorted_idx]
        num_options = num_options[sorted_idx]

        out = self.unroll_trajectory2(
            num_trajs=len(exp_trajs), max_length=self.args.max_length, batch_size=self.args.batch_size)
        learner_observations = out[0]
        learner_actions = out[1]
        learner_len = out[2]

        learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()))
        learner_act = np.zeros((learner_len.sum()))
        learner_l = np.zeros((learner_len.sum()))
        cnt = 0
        for i0 in range(learner_len.shape[0]):
            for j0 in range(1, learner_len[i0]+1):
                try:
                    learner_obs[cnt, :j0] = learner_observations[i0, :j0]
                    learner_act[cnt] = int(learner_actions[i0][j0-1])
                    learner_l[cnt] = j0
                    cnt += 1
                except:
                    break
        idx = learner_l != 0
        learner_obs = learner_obs[idx]
        learner_act = learner_act[idx]
        learner_l = learner_l[idx]
        learner_obs, learner_act, learner_len = arr_to_tensor(
            find_state, device, learner_obs, learner_act, learner_l)

        exp_obs, exp_act, exp_len = trajs_to_tensor(exp_trajs)
        exp_obs, exp_act, exp_len = arr_to_tensor(
            find_state, device, exp_obs, exp_act, exp_len)

        expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
        learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

        expert_loader = DataLoader(
            dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
        learner_loader = DataLoader(
            dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)

        expert_acc, learner_acc, discrim_loss = None, None, None

        for expert_data, learner_data in zip(enumerate(expert_loader), enumerate(learner_loader)):
            sampled_exp_obs, sampled_exp_len, sampled_exp_act = expert_data[1]
            sampled_exp_len, idxs = torch.sort(
                sampled_exp_len, descending=True)
            sampled_exp_obs = sampled_exp_obs[idxs]
            sampled_exp_act = sampled_exp_act[idxs]

            sampled_learner_obs, sampled_learner_len, sampled_learner_act = learner_data[1]
            sampled_learner_len, idxs = torch.sort(
                sampled_learner_len, descending=True)
            sampled_learner_obs = sampled_learner_obs[idxs]
            sampled_learner_act = sampled_learner_act[idxs]

            expert_acc, learner_acc, discrim_loss = \
                self.train_discrim_step(sampled_exp_obs.to(self.device),
                                        sampled_exp_act.to(self.device),
                                        sampled_exp_len.to(self.device),
                                        sampled_learner_obs.to(self.device),
                                        sampled_learner_act.to(self.device),
                                        sampled_learner_len.to(self.device)
                                        )

            return expert_acc, learner_acc, discrim_loss

    def pretrain_rnn(self, stateseq_in, stateseq_target, stateseq_len, num_options):
        out = self.Policy.pretrain_forward(stateseq_in, stateseq_len)
        criterion = torch.nn.NLLLoss(ignore_index=self.pad_idx)

        size0 = stateseq_target.size(0) * stateseq_target.size(1)

        out = out.view((size0, self.Policy.prob_dim))
        num_options = num_options.reshape((size0, ))
        stateseq_target = stateseq_target.view((size0,))

        idx = num_options != 0
        out = out[idx]
        num_options = num_options[idx]
        stateseq_target = stateseq_target[idx]

        mask1 = torch.zeros_like(out)
        for l0 in torch.unique(num_options):
            mask1[num_options == l0, :l0] = 1

        out = out + (mask1 + 1e-10).log()
        log_prob = torch.nn.functional.log_softmax(out, dim=1)

        rnnloss = criterion(input=log_prob,
                            target=stateseq_target)

        acc = torch.argmax(out, 1) == stateseq_target
        acc = acc.float()
        acc = torch.sum(acc) / acc.shape[0]

        prob = torch.softmax(out, 1)
        prob_idx = stateseq_target

        idxs = torch.arange(0, prob.shape[0]).to(self.device)
        acc2 = torch.mean(prob[idxs][idxs, prob_idx[idxs]])

        self.pretrain_opt.zero_grad()
        rnnloss.backward()
        self.pretrain_opt.step()

        return acc, acc2, rnnloss

    def train_discrim_step(self, exp_obs, exp_act, exp_len, learner_obs, learner_act, learner_len, train=True):
        expert = self.Discrim(exp_obs, exp_act.detach(), exp_len)
        learner = self.Discrim(learner_obs, learner_act.detach(), learner_len)

        expert_target = torch.zeros_like(expert)
        learner_target = torch.ones_like(learner)

        # cross-entropy
        discrim_loss = self.discrim_criterion(expert, expert_target) + \
            self.discrim_criterion(learner, learner_target)

        if train:
            # discriminator 求导，反向传播
            self.discrim_opt.zero_grad()
            discrim_loss.backward()
            self.discrim_opt.step()

        expert_acc = ((expert < 0.5).float()).mean()
        learner_acc = ((learner > 0.5).float()).mean()

        return expert_acc, learner_acc, discrim_loss

    def calculate_cum_value(self, learner_obs, learner_act, learner_len):
        """
        Calculate a value needed for policy training
        """
        rewards = self.Discrim.get_reward(
            learner_obs, learner_act, learner_len).squeeze(1)

        next_obs = self.pad_idx*torch.ones(
            size=(learner_obs.size(0), learner_obs.size(1) + 1),
            device=learner_obs.device, dtype=torch.long)
        next_obs[:, :learner_obs.size(1)] = learner_obs
        last_idxs = learner_obs[torch.arange(
            0, learner_obs.size(0)).long(), learner_len-1]
        next_idxs = self.nettransition[last_idxs, learner_act]
        next_obs[torch.arange(0, learner_obs.size(0)).long(),
                 learner_len] = next_idxs.to(self.device)

        next_len = learner_len + 1
        next_obs[next_obs == -1] = self.pad_idx
        next_act_prob = self.Policy.forward(next_obs, next_len)

        action_idxs = torch.Tensor(
            [i for i in range(self.Policy.action_dim)]).long().to(learner_obs.device)
        action_idxs = torch.cat(learner_obs.size(0)*[action_idxs.unsqueeze(0)])
        next_values = torch.cat(
            [self.Value(next_obs, action_idxs[:, i],  next_len) for i in [0, 1, 2]], dim=1)
        next_value = torch.sum(
            next_act_prob.probs[:, :3] * next_values, axis=1)

        cum_value = rewards + self.gamma * next_value
        return cum_value

    def train_policy_value(self, learner_obs, learner_len, learner_act, train=True):
        """
        Train policy net.
        Params:
            -learner_obs: [N * L * dim_obs_emb], generated trajs
            -learner_len: [N], lengths of the generated trajs
            -learner_act: [N], the next actions of the generated trajs
        """
        learner_act_prob = self.Policy.forward(learner_obs, learner_len)
        cum_value = self.calculate_cum_value(
            learner_obs, learner_act, learner_len)

        loss_policy = (cum_value.detach() *
                       learner_act_prob.log_prob(learner_act)).mean()

        val_pred = self.Value(learner_obs, learner_act, learner_len)
        loss_value = self.value_criterion(
            val_pred, cum_value.detach().view(val_pred.size()))
        entropy = learner_act_prob.entropy().mean()

        # construct computation graph for loss
        loss = loss_policy - self.c_1 * loss_value + self.c_2 * entropy
        loss = -loss

        if train:
            self.policy_opt.zero_grad()
            self.value_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()
            self.value_opt.step()

        return loss_policy, loss_value, entropy, loss

    def train(self, exp_obs, exp_act, exp_len, learner_obs, learner_act, learner_len, train_mode="value_policy"):
        """
        The basic training unit of TrajGAIL.
        Trains discriminator and generator for given update numbers.
        """
        self.Discrim.train()
        self.Policy.train()
        self.Value.train()

        self.summary_cnt += 1

        # 1. Train Discriminator
        # construct dataset
        expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
        learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

        expert_loader = DataLoader(
            dataset=expert_dataset, batch_size=self.args.batch_size, shuffle=True)
        learner_loader = DataLoader(
            dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)

        for _ in range(self.num_discrim_update):

            result = []
            for expert_data, learner_data in zip(enumerate(expert_loader), enumerate(learner_loader)):
                # expert_data[0]是一个index数字
                sampled_exp_obs, sampled_exp_len, sampled_exp_act = expert_data[1]

                # sort sampled expert trajs
                sampled_exp_len, idxs = torch.sort(
                    sampled_exp_len, descending=True)
                sampled_exp_obs = sampled_exp_obs[idxs]
                sampled_exp_act = sampled_exp_act[idxs]

                # sort sampled generated trajs
                sampled_learner_obs, sampled_learner_len, sampled_learner_act = learner_data[
                    1]

                sampled_learner_len, idxs = torch.sort(
                    sampled_learner_len, descending=True)
                sampled_learner_obs = sampled_learner_obs[idxs]
                sampled_learner_act = sampled_learner_act[idxs]

                expert_acc, learner_acc, discrim_loss = \
                    self.train_discrim_step(exp_obs=sampled_exp_obs.to(self.device),
                                            exp_act=sampled_exp_act.to(
                                                self.device),
                                            exp_len=sampled_exp_len.to(
                                                self.device),
                                            learner_obs=sampled_learner_obs.to(
                                                self.device),
                                            learner_act=sampled_learner_act.to(
                                                self.device),
                                            learner_len=sampled_learner_len.to(
                                                self.device)
                                            )

                result.append(
                    [expert_acc.detach(), learner_acc.detach(), discrim_loss.detach()])

        dloss = torch.cat([x[2].unsqueeze(0) for x in result], 0).mean().item()
        e_acc = torch.cat([x[0].unsqueeze(0) for x in result], 0).mean().item()
        l_acc = torch.cat([x[1].unsqueeze(0) for x in result], 0).mean().item()

        # 2. Train Generator
        for _ in range(self.num_gen_update):
            result = []
            for learner_data in enumerate(learner_loader):
                sampled_learner_obs, sampled_learner_len, sampled_learner_act = learner_data[
                    1]
                sampled_learner_len, idxs = torch.sort(
                    sampled_learner_len, descending=True)
                sampled_learner_obs = sampled_learner_obs[idxs]
                sampled_learner_act = sampled_learner_act[idxs]
                loss_policy, loss_value, entropy, loss = \
                    self.train_policy_value(sampled_learner_obs.to(self.device),
                                            sampled_learner_len.to(
                                                self.device),
                                            sampled_learner_act.to(self.device)
                                            )
                result.append(
                    [loss_policy.detach(), loss_value.detach(), entropy.detach(), loss.detach()])

        loss_policy = torch.cat([x[0].unsqueeze(0)
                                for x in result], 0).mean().item()
        loss_value = torch.cat([x[1].unsqueeze(0)
                               for x in result], 0).mean().item()
        entropy = torch.cat([x[2].unsqueeze(0)
                            for x in result], 0).mean().item()
        loss = torch.cat([x[3].unsqueeze(0) for x in result], 0).mean().item()

        return dloss, e_acc, l_acc, loss_policy, loss_value, entropy, loss

    def test(self, exp_obs, exp_act, exp_len, learner_obs, learner_act, learner_len, train_mode="value_policy"):

        self.Discrim.eval()
        self.Policy.eval()
        self.Value.eval()

        # self = GAILRNN
        # self.summary_cnt += 1
        # Train Discriminator

        expert_dataset = sequence_data(exp_obs, exp_len, exp_act)
        learner_dataset = sequence_data(learner_obs, learner_len, learner_act)

        expert_loader = DataLoader(dataset=expert_dataset,
                                   batch_size=self.args.batch_size, shuffle=True)
        learner_loader = DataLoader(
            dataset=learner_dataset, batch_size=self.args.batch_size, shuffle=True)

        for _ in range(self.num_discrim_update):

            result = []
            for expert_data, learner_data in zip(enumerate(expert_loader), enumerate(learner_loader)):
                sampled_exp_obs, sampled_exp_len, sampled_exp_act = expert_data[1]
                sampled_exp_len, idxs = torch.sort(
                    sampled_exp_len, descending=True)
                sampled_exp_obs = sampled_exp_obs[idxs]
                sampled_exp_act = sampled_exp_act[idxs]

                sampled_learner_obs, sampled_learner_len, sampled_learner_act = learner_data[
                    1]
                sampled_learner_len, idxs = torch.sort(
                    sampled_learner_len, descending=True)
                sampled_learner_obs = sampled_learner_obs[idxs]
                sampled_learner_act = sampled_learner_act[idxs]

                expert_acc, learner_acc, discrim_loss = \
                    self.train_discrim_step(exp_obs=sampled_exp_obs.to(self.device),
                                            exp_act=sampled_exp_act.to(
                                                self.device),
                                            exp_len=sampled_exp_len.to(
                                                self.device),
                                            learner_obs=sampled_learner_obs.to(
                                                self.device),
                                            learner_act=sampled_learner_act.to(
                                                self.device),
                                            learner_len=sampled_learner_len.to(
                                                self.device),
                                            train=False
                                            )

                result.append(
                    [expert_acc.detach(), learner_acc.detach(), discrim_loss.detach()])

        dloss = torch.cat([x[2].unsqueeze(0) for x in result], 0).mean()
        e_acc = torch.cat([x[0].unsqueeze(0) for x in result], 0).mean()
        l_acc = torch.cat([x[1].unsqueeze(0) for x in result], 0).mean()

        """
        self.summary_test.add_scalar(
            'loss/discrim', dloss.item(), self.summary_cnt)
        self.summary_test.add_scalar(
            'accuracy/expert', e_acc.item(), self.summary_cnt)
        self.summary_test.add_scalar(
            'accuracy/learner', l_acc.item(), self.summary_cnt)
        print("Testing >>> Expert: %.2f%% | Learner: %.2f%%" %
              (e_acc * 100, l_acc * 100))"""

    def save_model(self, outdir: str, iter0=0):
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        policy_statedict = self.Policy.state_dict()
        value_statedict = self.Value.state_dict()
        discrim_statedict = self.Discrim.state_dict()

        outdict = {"Policy": policy_statedict,
                   "Value": value_statedict,
                   "Discrim": discrim_statedict}

        data_split = []
        splited = self.args.data

        while True:
            splited = os.path.split(splited)
            if len(splited[1]) == 0:
                break
            else:
                data_split.append(splited[1])
                splited = splited[0]

        # self.args.data = "data/Binomial.csv"
        dataname = data_split[0].split('.')[0]
        filename = "ModelParam_"+dataname + "_"+str(iter0)+".pt"
        datapath = os.path.join(outdir, filename)
        print("model saved at {}".format(datapath))
        torch.save(outdict, datapath)

    def load_model(self, model_path):
        model_dict = torch.load(model_path)

        self.Policy.load_state_dict(model_dict['Policy'])
        print("Policy Model loaded Successfully")

        self.Value.load_state_dict(model_dict['Value'])
        print("Value Model loaded Successfully")

        self.Discrim.load_state_dict(model_dict['Discrim'])
        print("Discrim Model loaded Successfully")

    def unroll_batch(self, batch_size, max_length, origin_ratio=None):

        # generate one batch of trajectories
        def find_state(x): return self.env.states.index(x)

        np_find_state = np.vectorize(find_state)
        obs = np.ones((batch_size, 1), int) * self.env.start
        obs_len = np.ones((batch_size), np.int64)
        obs_len = torch.LongTensor(obs_len)
        done_mask = np.zeros_like(obs_len, bool)
        actions = np.zeros_like(obs)
        rewards = np.zeros(obs.shape)

        for i in range(max_length):
            # finished trajectories are masked
            notdone_obs = obs[~done_mask]
            notdone_obslen = obs_len[~done_mask]

            if notdone_obs.shape[0] == 0:
                break
            # find the state index
            state = np_find_state(notdone_obs)
            state = torch.LongTensor(state)

            action_dist = self.Policy.forward(state.to(self.device),
                                              notdone_obslen.to(self.device))

            if origin_ratio is not None:
                if i == 0:
                    origin_ratio = torch.Tensor(origin_ratio).unsqueeze_(0)
                    origin_ratio = torch.cat(
                        [origin_ratio for i in range(action_dist.probs.shape[0])], )
                    action_dist.probs = origin_ratio

            action = action_dist.sample()

            last_state = state.cpu().numpy()[:, -1]
            action = action.cpu().numpy()
            last_obs = np.array(self.env.states)[last_state]

            reward = self.Discrim.get_reward(state.to(self.device),
                                             torch.LongTensor(
                                                 action).to(self.device),
                                             notdone_obslen.to(self.device)
                                             )
            reward = reward.squeeze(1).cpu().numpy()

            def find_next_state(x): return self.env.netconfig[x[0]][x[1]]
            next_obs = np.apply_along_axis(
                find_next_state, 1, np.vstack([last_obs, action]).T)

            # prepare a tempt for next step of unfinished trajs
            unmasked_next_obs = -1 * np.ones_like(obs_len)
            unmasked_actions = -1 * np.ones_like(obs_len)
            unmasked_reward = -1 * np.ones(obs_len.shape)

            # fill the next steps in the tempt
            unmasked_next_obs[~done_mask] = next_obs
            unmasked_actions[~done_mask] = action
            unmasked_reward[~done_mask] = reward

            # concatenate the new step's action choice, state idx and rewards with historic steps
            obs = np.c_[obs, unmasked_next_obs]
            actions = np.c_[actions, unmasked_actions]
            rewards = np.c_[rewards, unmasked_reward]

            # get the mask for finished trajs
            is_done = np.vectorize(lambda x: (
                x == self.env.terminal) | (x == -1))
            done_mask = is_done(unmasked_next_obs)
            obs_len += 1-done_mask

        # remove the first step (from start-flag to entrances)
        actions = actions[:, 1:]
        rewards = rewards[:, 1:]

        # return the states (not state indexes), actions, traj lens, and rewards
        return obs, actions, obs_len.numpy(), rewards

    def unroll_trajectory2(self, *args, **kwargs):
        # generate certain number of trajs with a maximum length of max_length
        if kwargs:
            num_trajs = kwargs.get("num_trajs", self.args.pretrain_epoch_size)
            max_length = kwargs.get("max_length", self.args.max_length)
            batch_size = kwargs.get("batch_size", self.args.batch_size)
            origin_ratio = kwargs.get("origin_ratio", None)
        elif args:
            num_trajs, max_length, batch_size, origin_ratio = args
        else:
            raise Exception("wrong input")

        # containers for results
        obs = np.ones((num_trajs, max_length+1), np.int32) * -1
        actions = np.ones((num_trajs, max_length), np.int32) * -1
        obs_len = np.zeros((num_trajs,), np.int32)
        rewards = np.zeros((num_trajs, max_length))

        out_max_length = 0
        processed = 0

        for i in range(int(np.ceil(num_trajs / batch_size))):
            # generate one batch of trajs
            batch_obs, batch_act, batch_len, batch_reward = self.unroll_batch(
                batch_size, max_length, origin_ratio)

            batch_max_length = batch_obs.shape[1]
            if num_trajs - processed > batch_size:
                obs[(i*batch_size):((i+1)*batch_size),
                    :batch_max_length] = batch_obs
                actions[(i*batch_size):((i+1)*batch_size),
                        :(batch_max_length-1)] = batch_act
                obs_len[(i*batch_size):((i+1)*batch_size)] = batch_len
                rewards[(i*batch_size):((i+1)*batch_size),
                        :(batch_max_length-1)] = batch_reward
                processed += batch_obs.shape[0]
            else:
                obs[(i*batch_size):((i+1)*batch_size),
                    :batch_max_length] = batch_obs[:(num_trajs - processed)]
                actions[(i*batch_size):((i+1)*batch_size),
                        :(batch_max_length-1)] = batch_act[:(num_trajs - processed)]
                obs_len[(i*batch_size):((i+1)*batch_size)
                        ] = batch_len[:(num_trajs - processed)]
                rewards[(i*batch_size):((i+1)*batch_size), :(batch_max_length-1)
                        ] = batch_reward[:(num_trajs - processed)]

            out_max_length = max(out_max_length, batch_max_length)

        # remove the last step (from exits to end-flag)
        obs = obs[:, :out_max_length]
        actions = actions[:, :(out_max_length-1)]
        rewards = rewards[:, :(out_max_length-1)]

        return obs, actions, obs_len, rewards
