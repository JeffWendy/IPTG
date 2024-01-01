import os
import sys
sys.path.append(os.getcwd())

#!/usr/bin/python3
# from models.utils.plotutils import *
from utils import *
#from mdp import shortestpath
from trainer import GAILRNNTrain
from model import Discriminator as Discriminator_rnn
from model import Policy_net, Value_net
from utils import ShortestPath
import argparse
import time
import os
import numpy as np
import torch
import pickle as pkl

def argparser():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1000), type=int)
    parser.add_argument('--n-episode', default=int(640), type=int)
    parser.add_argument('--pretrain-step', default=int(0), type=int)
    parser.add_argument('--result-size', default=int(10000), type=int)
    parser.add_argument('--result-batch-size', default=int(1000), type=int)
    parser.add_argument('--pretrain-epoch-size', default=int(640), type=int)
    parser.add_argument('-b', '--batch-size', default=int(64), type=int)
    parser.add_argument('-nh', '--hidden', default=int(64), type=int)
    parser.add_argument('-ud', '--num-discrim-update',
                        default=int(2), type=int)
    parser.add_argument('-ug', '--num-gen-update', default=int(6), type=int)
    parser.add_argument('-lr', '--learning-rate',
                        default=float(5e-5), type=float)
    parser.add_argument('--c_1', default=float(1), type=float)
    parser.add_argument('--c_2', default=float(0.01), type=float)
    parser.add_argument('--eps', default=float(1e-6), type=float)
    parser.add_argument('--cuda', default=True, type=bool)
    parser.add_argument('--train-mode', default="value_policy", type=str)
    parser.add_argument('--data', default="./baselines/TrajGAIL/data/Expert_demo.csv", type=str)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('-w', '--wasser', action="store_true", default=False)
    parser.add_argument('-ml', '--max-length', default=int(60), type=int)
    return parser.parse_args()


args = argparser()

# 构建networks
def main(args):

    """check in and outs"""
    netins = [0, 24, 16, 47]
    netouts = [9, 23, 15, 65]
    """ check path """
    env = ShortestPath("./baselines/TrajGAIL/data/Network.txt", netins, netouts)

    exp_trajs = env.import_demonstrations(args.data)
    pad_idx = len(env.states)

    if torch.cuda.is_available() & args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ob_space = env.n_states
    act_space = env.max_actions

    state_dim = ob_space
    action_dim = act_space

    def find_state(x): return env.states.index(x) if x != -1 else pad_idx
    find_state = np.vectorize(find_state)

    origins = np.array(env.origins)
    origins = find_state(origins)
    origins = torch.Tensor(origins).long().to(device)

    policy = Policy_net(state_dim, action_dim,
                        hidden=args.hidden,
                        origins=origins,
                        start_code=env.states.index(env.start),
                        env=env,
                        disttype="categorical")

    value = Value_net(
        state_dim, action_dim, hidden=args.hidden, num_layers=args.num_layers)

    D = Discriminator_rnn(
        state_dim, action_dim, hidden=args.hidden, disttype="categorical", num_layers=args.num_layers)

    if device.type == "cuda":
        policy = policy.cuda()
        value = value.cuda()
        D = D.cuda()

    GAILRNN = GAILRNNTrain(env=env,
                           Policy=policy,
                           Value=value,
                           Discrim=D,
                           pad_idx=pad_idx,
                           args=args)
    GAILRNN.set_device(device)
    if args.wasser:
        GAILRNN.train_discrim_step = GAILRNN.train_wasser_discrim_step
        GAILRNN.discrim_opt = torch.optim.RMSprop(
            GAILRNN.Discrim.parameters(), lr=GAILRNN.lr, eps=GAILRNN.eps)

    if args.pretrain_step > 0:
        print("********* PRETRAIN STARTED *********\n")

    for i in range(args.pretrain_step):
        st_time = time.time()
        st_idx = (i * args.pretrain_epoch_size) % len(exp_trajs)
        end_idx = ((i + 1) * args.pretrain_epoch_size) % len(exp_trajs)
        st_idx = min(st_idx, end_idx)
        end_idx = max(end_idx, st_idx)
        acc, acc2, loss, expert_acc, learner_acc, discrim_loss = GAILRNN.pretrain(
            exp_trajs[st_idx:end_idx], find_state, device)
        end_time = time.time()
        elapsed = end_time - st_time
        if i % 5 == 0:
            print("\nEpoch {:d}, {:.5f} seconds elapsed processing this epoch\n".format(i, elapsed) +
                "\tgenerator: acc = {:.5f}%,  acc2 = {:.5f}%,  loss = {:.5f}\n".format(acc*100, acc2*100, loss) +
                "\tdiscriminator: acc = {:.5f}%,  acc2 = {:.5f}%, loss = {:.5f}".format(expert_acc*100, learner_acc*100, discrim_loss))

    hard_update(GAILRNN.Value.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)
    hard_update(GAILRNN.Discrim.StateSeqEmb, GAILRNN.Policy.StateSeqEmb)

    print("********* TRAINING STARTED *********\n")

    for i in range(args.iteration):
        if i % 5 == 0:
            print("Epoch {:d}:".format(i))
        now = time.time()
        learner_observations, learner_actions, learner_len, learner_rewards =\
            GAILRNN.unroll_trajectory2(
                num_trajs=args.n_episode, max_length=args.max_length)

        mask = learner_rewards != -1
        if i % 5 == 0:
            avg_reward = np.mean(np.sum(learner_rewards * mask, axis=1))
            avg_ind_reward = (learner_rewards * mask).sum() / mask.sum()
            avg_len = learner_len.mean()
            print("\tavg reward: {:.5f}, avg_ind_reward:{:.5f}, avg_len:{:.5f}".
                format(avg_reward, avg_ind_reward, avg_len))

        learner_obs = -1 * np.ones((learner_len.sum(), learner_len.max()))
        learner_act = np.zeros((learner_len.sum()))
        learner_l = np.zeros((learner_len.sum()))
        cnt = 0
        for i0 in range(learner_len.shape[0]):
            for j0 in range(1, learner_len[i0]+1):
                """
                learner_obs[cnt, :j0] = learner_observations[i0, :j0]
                learner_act[cnt] = int(learner_actions[i0][j0-1])
                learner_l[cnt] = j0
                cnt += 1
                """
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
        learner_obs, learner_act, learner_len = arr_to_tensor(find_state, device, learner_obs, learner_act, learner_l)

        sample_indices = np.random.randint(
            low=0, high=len(exp_trajs), size=args.n_episode)
        exp_trajs_temp = np.take(a=exp_trajs, indices=sample_indices, axis=0)
        exp_obs, exp_act, exp_len = trajs_to_tensor(exp_trajs_temp)
        exp_obs, exp_act, exp_len = arr_to_tensor(
            find_state, device, exp_obs, exp_act, exp_len)

        dloss, e_acc, l_acc, loss_policy, loss_value, entropy, loss = GAILRNN.train(
                    exp_obs=exp_obs,
                    exp_act=exp_act,
                    exp_len=exp_len,
                    learner_obs=learner_obs,
                    learner_act=learner_act,
                    learner_len=learner_len)

        elapsed = time.time() - now
        
        if i % 5 == 0:
            print("\tdiscriminator: dloss = {:.5f},  e_acc = {:.5f},  l_acc = {:.5f}".format(dloss, e_acc, l_acc))
            print("\tpolicy and value: ploss = {:.5f},  vloss = {:.5f},  pentropy = {:.5f}, loss = {:.5f}"
                .format(loss_policy, loss_value, entropy, loss))
            print("\t{} seconds elapsed when processing this epoch".format(elapsed))
        
        if i % 10 == 0:
            x = None
            for j in range(int((args.result_size * 1.0 / args.result_batch_size))):
                tmp = GAILRNN.unroll_trajectory2(num_trajs=args.result_batch_size, max_length=args.max_length, batch_size=args.batch_size)
                if j == 0:
                    x = tmp[0]
                else:
                    x = np.concatenate([x, tmp[0]], axis=0)
            with open("./baselines/TrajGAIL/results/result{}".format(i), "wb") as f:
                pkl.dump(x, f)
            torch.save(policy.state_dict(), './baselines/TrajGAIL/params/P_param{}'.format(i))
            torch.save(value.state_dict(), './baselines/TrajGAIL/params/V_param{}'.format(i))
            torch.save(D.state_dict(), './baselines/TrajGAIL/params/D_param{}'.format(i))

if __name__ == '__main__':
    args = argparser()
    main(args)