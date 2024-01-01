import numpy as np
import math
from collections import namedtuple
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))

"""
class model_summary_writer(object):
    def __init__(self, summary_name, env):
        now = datetime.datetime.now()
        self.summary = SummaryWriter(
            logdir='log/' + summary_name + '_{}'.format(now.strftime('%Y%m%d_%H%M%S')))
        self.summary_cnt = 0
        self.env = env
"""


class sequence_data(Dataset):
    def __init__(self, obs, len0, act):
        self.obs = obs
        self.len = len0
        self.act = act

        self.data_size = obs.size(0)

    def __getitem__(self, index):
        return self.obs[index], self.len[index], self.act[index]

    def __len__(self):
        return self.data_size


class sequence_data_vanilla(Dataset):
    def __init__(self, obs, act):
        self.obs = obs
        self.act = act

        self.data_size = obs.size(0)

    def __getitem__(self, index):
        return self.obs[index], self.act[index]

    def __len__(self):
        return self.data_size


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1, 1)


def get_gae(rewards, learner_len, values, gamma, lamda):
    # rewards = learner_rewards[1]
    # learner_len=learner_len[1]
    # values = learner_values[1]
    # gamma = args.gamma
    # lamda = args.lamda

    rewards = torch.Tensor(rewards)
    returns = torch.zeros_like(rewards)
    advants = -1 * torch.ones_like(rewards)

    masks = torch.ones_like(rewards)
    masks[(learner_len-1):] = 0

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, learner_len)):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        running_tderror = rewards[t] + gamma * \
            previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + gamma * \
            lamda * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants[:learner_len] = (
        advants[:learner_len] - advants[:learner_len].mean()) / advants[:learner_len].std()
    return returns, advants


def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def unsqueeze_trajs(learner_observations, learner_actions, learner_len):
    learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()+1))
    learner_act = np.zeros((learner_len.sum()))
    learner_l = np.zeros((learner_len.sum()))
    cur_idx = 0
    for length0 in range(1, np.max(learner_len)):
        takeidxs = np.where(learner_len >= length0)[0]
        learner_obs[cur_idx:(cur_idx + takeidxs.shape[0]),
                    :length0] = learner_observations[takeidxs, :length0]
        learner_act[cur_idx:(cur_idx + takeidxs.shape[0])
                    ] = learner_actions[takeidxs, (length0-1)]
        learner_l[cur_idx:(cur_idx+takeidxs.shape[0])] = length0
        cur_idx += takeidxs.shape[0]

    idx = learner_l != 0
    learner_obs = learner_obs[idx]
    learner_act = learner_act[idx]
    learner_l = learner_l[idx]

    idx = learner_act != -1
    learner_obs = learner_obs[idx]
    learner_act = learner_act[idx]
    learner_l = learner_l[idx]

    return learner_obs, learner_act, learner_l


def trajs_squeezedtensor(exp_trajs):
    exp_obs = [[x.cur_state for x in episode]+[episode[-1].next_state]
               for episode in exp_trajs]
    exp_act = [[x.action for x in episode] for episode in exp_trajs]
    exp_len = np.array(list(map(len, exp_obs)))

    max_len = max(exp_len)

    expert_observations = np.ones((len(exp_trajs), max_len), np.int32) * -1
    expert_actions = np.ones((len(exp_trajs), max_len-1), np.int32) * -1

    for i in range(len(exp_obs)):
        expert_observations[i, :exp_len[i]] = exp_obs[i]
        expert_actions[i, :(exp_len[i]-1)] = exp_act[i]

    return expert_observations, expert_actions, exp_len


def trajs_to_tensor(exp_trajs):
    np_trajs = []
    for episode in exp_trajs:
        for i in range(1, len(episode)+1):
            # every element is composed of an array of steps and the next action of that array
            np_trajs.append(
                [[x.cur_state for x in episode[:i]], episode[i-1].action])

    expert_len = np.array([len(x[0]) for x in np_trajs])
    maxlen = np.max(expert_len)

    expert_observations = -np.ones(shape=(len(np_trajs), maxlen))
    expert_actions = np.array([x[1] for x in np_trajs])

    expert_len = []
    for i in range(len(np_trajs)):
        temp = np_trajs[i][0]
        expert_observations[i, :len(temp)] = temp
        expert_len.append(len(temp))

    return expert_observations, expert_actions, expert_len


def arr_to_tensor(find_state, device, exp_obs, exp_act, exp_len):
    exp_states = find_state(exp_obs)
    exp_obs = torch.LongTensor(exp_states)
    exp_act = torch.LongTensor(exp_act)
    exp_len = torch.LongTensor(exp_len)
    # exp_len , sorted_idx = exp_len.sort(0,descending = True)
    # exp_obs = exp_obs[sorted_idx]
    # exp_act = exp_act[sorted_idx]
    return exp_obs, exp_act, exp_len


Step = namedtuple('Step', 'cur_state action next_state reward done')


def check_RouteID(episode, routes):
    state_seq = [str(x.cur_state) for x in episode] + \
        [str(episode[-1].next_state)]
    episode_route = "-".join(state_seq)
    if episode_route in routes:
        idx = routes.index(episode_route)
    else:
        idx = -1
    return idx


def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


def sigmoid(xs):
    """
    sigmoid function
    inputs:
      xs      1d array
    """
    return [1 / (1 + math.exp(-x)) for x in xs]


def identify_routes(trajs):
    num_trajs = len(trajs)
    route_dict = {}
    for i in range(num_trajs):
        episode = trajs[i]
        route = "-".join([str(x.cur_state)
                          for x in episode] + [str(episode[-1].next_state)])
        if route in route_dict.keys():
            route_dict[route] += 1
        else:
            route_dict[route] = 1

    out_list = []
    for key in route_dict.keys():
        route_len = len(key.split("-"))
        out_list.append((key, route_len, route_dict[key]))
    out_list = sorted(out_list, key=lambda x: x[2],  reverse=True)
    return out_list


def expert_compute_state_visitation_freq(sw, trajs):
    feat_exp = np.zeros([sw.n_states])
    for episode in trajs:
        for step in episode:
            feat_exp[sw.pos2idx(step.cur_state)] += 1
        feat_exp[sw.pos2idx(step.next_state)] += 1
    feat_exp = feat_exp/len(trajs)
    return feat_exp


def expert_compute_state_action_visitation_freq(sw, trajs):
    N_STATES = sw.n_states
    N_ACTIONS = sw.max_actions

    mu = np.zeros([N_STATES, N_ACTIONS])

    for episode in trajs:
        for step in episode:
            cur_state = step.cur_state
            s = sw.pos2idx(cur_state)
            action_list = sw.get_action_list(cur_state)
            action = step.action
            a = action_list.index(action)
            mu[s, a] += 1

    mu = mu/len(trajs)
    return mu


def compute_state_visitation_freq(sw, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """
    N_STATES = sw.n_states
    # N_ACTIONS = sw.max_actions

    T = len(trajs[0])+1
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for traj in trajs:
        mu[sw.pos2idx(traj[0].cur_state), 0] += 1
    mu[:, 0] = mu[:, 0]/len(trajs)

    for t in range(T-1):
        for s in range(N_STATES):
            if deterministic:
                mu[s, t+1] = sum([mu[pre_s, t]*sw.is_connected(sw.idx2pos(pre_s), np.argmax(
                    policy[pre_s]), sw.idx2pos(s)) for pre_s in range(N_STATES)])
                # mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, np.argmax(policy[pre_s])] for pre_s in range(N_STATES)])
            else:
                mu_temp = 0
                for pre_s in range(N_STATES):
                    action_list = sw.get_action_list(sw.idx2pos(pre_s))
                    for a1 in range(len(action_list)):
                        mu_temp += mu[pre_s, t]*sw.is_connected(sw.idx2pos(
                            pre_s), action_list[a1], sw.idx2pos(s)) * policy[pre_s, a1]

                mu[s, t+1] = mu_temp
                # mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])

    p = np.sum(mu, 1)
    return p


def compute_state_action_visitation_freq(sw, gamma, trajs, policy, deterministic=True):
    """compute the expected states visition frequency p(s| theta, T) 
    using dynamic programming

    inputs:
      P_a     NxNxN_ACTIONS matrix - transition dynamics
      gamma   float - discount factor
      trajs   list of list of Steps - collected from expert
      policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy


    returns:
      p       Nx1 vector - state visitation frequencies
    """

    N_STATES = sw.n_states
    N_ACTIONS = sw.max_actions

    route_list = identify_routes(trajs)
    max_route_length = max([x[1] for x in route_list])
    T = max_route_length

    mu = np.zeros([N_STATES, N_ACTIONS, T])
    start_state = sw.start
    s = sw.pos2idx(start_state)
    action_list = sw.get_action_list(start_state)
    mu[s, :, 0] = policy[s, :]

    for t in range(T-1):
        for s in range(N_STATES):
            action_list = sw.get_action_list(sw.idx2pos(s))
            for a in range(len(action_list)):
                next_state = sw.netconfig[sw.idx2pos(s)][action_list[a]]
                s1 = sw.pos2idx(next_state)
                next_action_list = sw.get_action_list(next_state)
                mu[s1, :len(next_action_list), t+1] += mu[s, a, t] * \
                    policy[s1, :len(next_action_list)]

    p = np.sum(mu, axis=2)
    # p.shape
    return p


def atanh(x):
    x = torch.clamp(x, -1+1e-7, 1-1e-7)
    out = torch.log(1+x) - torch.log(1-x)
    return 0.5*out


class ShortestPath(object):
    """

    ShortestPath Algorithm Imitation Learning Environment

    """

    def __init__(self, network_path, origins, destinations):
        self.network_path = network_path
        self.netin = origins
        self.netout = destinations

        try:
            netconfig = pd.read_csv(os.path.join(os.path.dirname(
                os.getcwd()), self.network_path), sep=" ", header=None)
        except:
            netconfig = pd.read_csv(self.network_path, sep=" ", header=None)

        netconfig = netconfig[[0, 1, 2]]
        netconfig.columns = ["from", "con", "to"]
        # 1: right-turn 2: straight 3: left-turn

        # netconfig_dict:
        # {
        #   from1 : {
        #     [con1(1) : to1,]
        #     [con2(2) : to2,]
        #     [con3(3) : to3,]
        #     4 : terminal(1001)
        #   }
        #   start(1000) : {
        #     i: orgini (state1, e.g. any road segment)
        #   }
        # }
        #
        netconfig_dict = {}
        for i in range(len(netconfig)):
            fromid, con, toid = netconfig.loc[i]

            if fromid in netconfig_dict.keys():
                netconfig_dict[fromid][con] = toid
            else:
                netconfig_dict[fromid] = {}
                netconfig_dict[fromid][con] = toid
        self.netconfig = netconfig_dict

        # states : all the road segemnts (id)
        states = list(set(list(netconfig["from"]) + list(netconfig["to"])))
        states += [x for x in destinations if x not in states]
        states += [x for x in origins if x not in states]

        """check start terminal flag"""
        self.start = 2751
        self.terminal = 2752

        self.origins = origins
        self.destinations = destinations
        self.max_actions = 1338

        self.actions = [i for i in range(self.max_actions)]
        self.action_end = 1333
        #
        self.netconfig[self.start] = {}
        for i in range(len(self.origins)):
            self.netconfig[self.start][self.action_end + 1 + i] = self.origins[i]
        for d0 in self.destinations:
            if d0 not in self.netconfig.keys():
                self.netconfig[d0] = {}

            """check terminal action idx"""
            self.netconfig[d0][self.action_end] = self.terminal
        
        # states: all the road segemnts, and start/end flag
        states = states + [self.start, self.terminal]
        # 0 = start of the trip
        # 1 = end of the trip

        self.states = states
        """check action list"""
        # 1: right-turn 2: straight 3: left-turn 4:end_trip

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)

        # max_actions: the maximum count of actions that a state can lead to
        #self.max_actions = max([len(self.get_action_list(s))
        #                       for s in self.netconfig.keys()])


        # rewards: a list that equals to states in length
        self.rewards = [0 for i in range(self.n_states)]

        # state_action_pair: [(state1, action1), (state1, action2), ... ...]
        self.state_action_pair = sum([[(s, a) for a in self.get_action_list(
            s)] for s in self.states], [])  # add lists, i.e. concate lists
        self.num_sapair = len(self.state_action_pair)

        # sapair_idxs : a list of indexes of state and action pairs, probably for look-up
        self.sapair_idxs = []
        for i in range(len(self.state_action_pair)):
            state, action = self.state_action_pair[i]
            s = self.pos2idx(state)
            action_list = self.get_action_list(state)
            a = action_list.index(action)
            self.sapair_idxs.append((s, a))

        self.policy_mask = np.zeros([self.n_states, self.max_actions])
        for s, a in self.sapair_idxs:
            self.policy_mask[s, a] = 1

        # # for d0 in self.destinations:
        #   # self.rewards[self.pos2idx(d0)] = 1

        # self.rewards[self.pos2idx(252)] = 1
        # self.rewards[self.pos2idx(442)] = 1
        # self.rewards[self.pos2idx(1)] = 1

    def pos2idx(self, state):
        """
        input:
          state id
        returns:
          id index
        """
        return self.states.index(state)

    def idx2pos(self, idx):
        """
        input:
          id idx
        returns:
          state id
        """
        return self.states[idx]

    # def __init__(self, rewards, terminals, move_rand=0.0):
    #   """
    #   inputs:
    #     rewards     1d float array - contains rewards
    #     terminals   a set of all the terminal states
    #   """
    #   self.n_states = len(rewards)
    #   self.rewards = rewards
    #   self.terminals = terminals
    #   self.actions = [-1, 1]
    #   self.n_actions = len(self.actions)
    #   self.move_rand = move_rand

    def get_reward(self, state):
        return self.rewards[self.pos2idx(state)]

    # get the next state(road segment) that the "action" on the current state leads to
    def get_state_transition(self, state, action):
        return self.netconfig[state][action]

    def get_action_list(self, state):
        if state in self.netconfig.keys():
            return list(self.netconfig[state].keys())
        else:
            return list()

    def get_transition_states_and_probs(self, state, action):
        """
        inputs: 
          state       int - state
          action      int - action

        returns
          a list of (state, probability) pair
        """
        return [(self.netconfig[state][action], 1)]

    def is_connected(self, state, action, state1):
        try:
            return self.netconfig[state][action] == state1
        except:
            return False

        # if state == self.start:
        #   return [(self.origins[action] , 1)]
        # elif state in self.destinations:
        #   if action == 2:
        #     return [(self.terminal, 1)]
        #   else:
        #     return None
        # else:
        #   return [(self.netconfig[state][action] , 1)]

    def is_terminal(self, state):
        if state == self.terminal:
            # if state in self.destinations:
            return True
        else:
            return False

    ##############################################
    # Stateful Functions For Model-Free Leanring #
    ##############################################

    # reset current state to start flag
    def reset(self, start_pos=0):
        self._cur_state = start_pos

    def get_current_state(self):
        return self._cur_state

    def step(self, action):
        """
        Step function for the agent to interact with gridworld
        inputs: 
          action        action taken by the agent
        returns
          current_state current state
          action        input action
          next_state    next_state
          reward        reward on the next state
          is_done       True/False - if the agent is already on the terminal states
        """
        if self.is_terminal(self._cur_state):
            self._is_done = True
            return self._cur_state, action, self._cur_state, self.get_reward(self._cur_state), True

        next_state = self.get_state_transition(self._cur_state, action)
        last_state = self._cur_state

        reward = self.get_reward(last_state)
        self._cur_state = next_state

        if self.is_terminal(next_state):
            self._is_done = True
            return last_state, action, next_state, reward, True
        return last_state, action, next_state, reward, False

    #######################
    # Some util functions #
    #######################

    def get_transition_mat(self):
        """
        get transition dynamics of the gridworld

        return:
          P_a         NxNxN_ACTIONS transition probabilities matrix - 
                        P_a[s0, s1, a] is the transition prob of 
                        landing at state s1 when taking action 
                        a at state s0
        """
        N_STATES = self.n_states
        N_ACTIONS = len(self.actions)
        P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))

        for si in range(N_STATES):
            for a in range(N_ACTIONS):
                # sj = self.get_state_transition(self.idx2pos(si) , self.actions[a])
                # P_a[si,sj,a]  = 1

                if not self.idx2pos(si) == self.terminal:
                    try:
                        probs = self.get_transition_states_and_probs(
                            self.idx2pos(si), self.actions[a])
                        for sj, prob in probs:
                            # Prob of si to sj given action a\
                            P_a[si, self.pos2idx(sj), a] = prob
                    except:
                        # 1
                        print(str(si) + " " + str(a))
        return P_a

    def generate_demonstrations(self, policy, n_trajs=100, len_traj=20):
        """gatheres expert demonstrations

        inputs:
        gw          Gridworld - the environment
        policy      Nx1 matrix
        n_trajs     int - number of trajectories to generate
        rand_start  bool - randomly picking start position or not
        start_pos   2x1 list - set start position, default [0,0]
        returns:
        trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
        """

        trajs = []
        # right,wrong = 0,0
        cnt = 0
        for i in range(n_trajs):
            try:
                episode = []
                self.reset(self.start)
                cur_state = self.start

                # action_list = list(self.netconfig[self._cur_state].keys())
                # action_prob = policy[self.pos2idx(self._cur_state)]
                # action_idx = np.random.choice(range(len(self.action_list)) , p=action_prob)
                # action = action_list[action_idx]
                # cur_state, action, next_state, reward, is_done = self.step(action)
                # episode.append(Step(cur_state=cur_state, action=action, next_state=next_state, reward=reward, done=is_done))
                # # while not is_done:
                for _ in range(1, len_traj):
                    if self.is_terminal(self._cur_state):
                        break
                    else:
                        action_list = list(
                            self.netconfig[self._cur_state].keys())
                        action_prob = policy[self.pos2idx(self._cur_state)]
                        action_idx = np.random.choice(
                            range(len(action_list)), p=action_prob[:len(action_list)])
                        action = action_list[action_idx]
                        cur_state, action, next_state, reward, is_done = self.step(
                            action)
                        episode.append(Step(cur_state=cur_state, action=action,
                                       next_state=next_state, reward=reward, done=is_done))
                trajs.append(episode)
            except:
                cnt += 1
                print("error count : " + str(cnt) + " / " + str(i))
        return trajs

    def import_demonstrations(self, demopath):
        if demopath.split(".")[-1] == "pkl":
            import pickle
            trajs = pickle.load(open(demopath, 'rb'))
            route_list = identify_routes(trajs)
            max_route_length = max([x[1] for x in route_list])
            self.max_route_length = max_route_length
            return trajs

        elif demopath.split(".")[-1] == "csv":
            demo = pd.read_csv(demopath)

            trajs = []
            error_count = 0

            """the oid of each traj must be unique"""
            oid_list = list(set(list(demo["oid"])))
            n_trajs = len(oid_list)

            for i in range(n_trajs):
                cur_demo = demo[demo["oid"] == oid_list[i]]
                cur_demo = cur_demo.reset_index()

                len_demo = cur_demo.shape[0]
                episode = []

                self.reset(self.start)
                
                try:
                    for i0 in range(len_demo):
                        _cur_state = self._cur_state
                        _next_state = cur_demo.loc[i0, "stayId"]

                        action_list = self.get_action_list(_cur_state)
                        j = [self.get_state_transition(
                            _cur_state, a0) for a0 in action_list].index(_next_state)

                        action = action_list[j]

                        cur_state, action, next_state, reward, is_done = self.step(
                            action)
                        episode.append(Step(cur_state=cur_state, action=action,
                                            next_state=next_state, reward=reward, done=is_done))

                    cur_state, action, next_state, reward, is_done = self.step(
                        self.action_end)
                    episode.append(Step(cur_state=cur_state, action=action,
                                        next_state=next_state, reward=reward, done=is_done))
                    trajs.append(episode)
                
                except Exception as e:
                    error_count += 1
                    #print(e)

            route_list = identify_routes(trajs)
            max_route_length = max([x[1] for x in route_list])
            self.max_route_length = max_route_length
            print("{} expert trajs loaded, error count: {}".format(len(trajs), error_count))
            return trajs
