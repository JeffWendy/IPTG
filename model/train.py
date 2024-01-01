import argparse
import pickle as pkl
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
from data_iter import DatasetFactory
from generator import Generator
from dloss import DLoss
from topoloss import TOPOLoss
from evaluation.evaluator import Evaluator
import time
import math

# Arguemnts
parser = argparse.ArgumentParser(description='TrajGEN')
parser.add_argument('--hpc', action='store_true', default=False,
                    help='set to hpc mode')
parser.add_argument('--data-path', type=str, default='./data/generated/', metavar='R',
                    help='dir to save generated samples')
parser.add_argument('--train_steps', type=int, default=50, metavar='N',
                    help='rounds of  training (default: 50)')
parser.add_argument('--pretrain_steps', type=int, default=80, metavar='N',
                    help='steps of pre-training of generators (default: 80)')
parser.add_argument('--adv_steps', type=int, default=1, metavar='N',
                    help='steps of generator updates in one round of adverarial training (default: 3)')
parser.add_argument('--mle_steps', type=int, default=3, metavar='N',
                    help='steps of discriminator updates in one round of adverarial training (default: 3)')
parser.add_argument('--gk_epochs', type=int, default=1, metavar='N',
                    help='epochs of generator updates in one step of generate update (default: 2)')
parser.add_argument('--dk_epochs', type=int, default=1, metavar='N',
                    help='epochs of discriminator updates in one step of discriminator update (default: 1)')
parser.add_argument('--update_rate', type=float, default=0.9, metavar='UR',
                    help='update rate of roll-out model (default: 0.9)')
parser.add_argument('--n_rollout', type=int, default=6, metavar='N',
                    help='number of roll-out (default: 6)')
parser.add_argument('--vocab_size', type=int, default=2753, metavar='N',
                    help='vocabulary size (default: 10)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--n_samples', type=int, default=6400, metavar='N',
                    help='number of samples gerenated per time (default: 6400)')
parser.add_argument('--gen_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of generator optimizer (default: 2e-3)')
parser.add_argument('--dis_lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate of discriminator optimizer (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ad_g_batch', type=int, default=100, metavar='R',
                    help=' generator training batch number')
parser.add_argument('--alpha', type=float, default=0.0, metavar='R',
                    help=' loss coefficient')
parser.add_argument('--beta', type=float, default=1e5, metavar='R',
                    help='topo_loss coefficient')
parser.add_argument('--gamma', type=float, default=1, metavar='R',
                    help='d_loss coefficient')
parser.add_argument('--interval', type=int, default=5, metavar='R',
                    help='evaluation interval')
parser.add_argument('--melting', type=int, default=0, metavar='R',
                    help='melting options; 0: Full, 1: No D, 2: No T, 3: No constraints, 4: No mask, 5: No mask or constriants')
# Files
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'

# Parse arguments
ARGS = parser.parse_args()
ARGS.cuda = not ARGS.no_cuda and torch.cuda.is_available()
ARGS.g_embed_dim = 32
ARGS.g_hidden_dim = 32
ARGS.g_state_embed_dim = 8
ARGS.dataset_size = 6400
torch.manual_seed(ARGS.seed)
if ARGS.cuda:
    torch.cuda.manual_seed(ARGS.seed)
if not ARGS.hpc:
    ARGS.data_path = ''
POSITIVE_FILE = ARGS.data_path + POSITIVE_FILE
NEGATIVE_FILE = ARGS.data_path + NEGATIVE_FILE
ARGS.num_batches = 150
ARGS.dt_epochs = 1
ARGS.mle_epochs = 1
ARGS.max_len = 35
ARGS.device = torch.device('cuda:0')
ARGS.seed_generator = torch.Generator(device=ARGS.device)
ARGS.seed_generator.manual_seed(3)

with open("./data/lens_distribution.pkl", "rb") as f:
    ARGS.lens_dist = pkl.load(f, encoding="iso-8859-1").to(ARGS.device).to(torch.float)

with open("./data/node_dist.pkl", "rb") as f:
    ARGS.node_dist = pkl.load(f, encoding="iso-8859-1")
    for seq_len in ARGS.node_dist:
        ARGS.node_dist[seq_len] = ARGS.node_dist[seq_len].to(ARGS.device)

with open("./data/one_step_accessiblity_matrix_.pkl", "rb") as f:
    ARGS.acc_mat = pkl.load(f, encoding="iso-8859-1")
    ARGS.acc_mat_numpy = ARGS.acc_mat.numpy()
    ARGS.acc_mat = ARGS.acc_mat.to(ARGS.device)

ARGS.SOT = ARGS.vocab_size -2
ARGS.EOT = ARGS.vocab_size -1
# Set models, criteria, optimizers
NLL_LOSS = nn.NLLLoss()

D_LOSS = DLoss()
TOPO_LOSS = TOPOLoss()
if ARGS.cuda:
    NLL_LOSS = NLL_LOSS.cuda()
    cudnn.benchmark = True

# Container of experiment data
GEN_PRETRAIN_TRAIN_LOSS = []
GEN_PRETRAIN_EVAL_LOSS = []
GEN_EVAL_LOSS = []

DATASET_FACTORY = DatasetFactory('real.data', None, ARGS)
SUB_GENERATORS_PATHS = []
PARENT_GENERATOR_PATH = './data/base_generator.pkl'

def multinomial(dist, num):
    return torch.multinomial(dist, num, replacement=True, generator=ARGS.seed_generator).squeeze()

def generate_samples(generator, n_trajs, path):
    samples = {}
    n_batches = round(n_trajs * 1.0 / ARGS.batch_size)
    for _ in range(n_batches):
        seq_len = multinomial(ARGS.lens_dist, 1).item()

        if seq_len not in samples:
            samples[seq_len] = []
        sample, _, _ = generator.sample(ARGS.batch_size, seq_len)
        sample = sample.cpu().data.numpy().tolist()
        samples[seq_len] += sample

    with open(path, 'w') as fout:
        for seq_len in samples:
            sample = samples[seq_len]
            fout.write('{}\n'.format(str(seq_len)))
            for seq in sample:
                string = ' '.join([str(s) for s in seq])
                fout.write('{}\n'.format(string))

    t = math.floor(time.time())
    """
    with open(str(t) + output_file, 'w') as fout:
        for seq_len in samples:
            sample = samples[seq_len]
            fout.write('{}\n'.format(str(seq_len)))
            for seq in sample:
                string = ' '.join([str(s) for s in seq])
                fout.write('{}\n'.format(string))
    """
    return samples

def eval_generator(generator, dataset):
    """
    Evaluate generator with NLL
    """
    total_loss = 0.
    with torch.no_grad():
        for data, target in dataset:
            if ARGS.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = generator(data)
            loss = NLL_LOSS(pred, target)
            total_loss += loss.item()
    avg_loss = total_loss / dataset.num_batches
    return avg_loss


def train_generator_MLE(dataset, generator, optimizer):
    for i in range(ARGS.mle_epochs):
        total_loss = 0.
        for data, target in dataset:
            if ARGS.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = generator(data)
            loss = NLL_LOSS(output, target)
            total_loss += loss.detach().to('cpu').numpy().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    avg_loss = total_loss / dataset.num_batches
    GEN_PRETRAIN_TRAIN_LOSS.append(avg_loss)
    return avg_loss
    
    """
    samples = generate_samples(generator, args.batch_size, args.num_batches, NEGATIVE_FILE, args.lens_dist)
    
    eval_iter = GenDataIter(POSITIVE_FILE, args.batch_size, args)
    # todo: 修改eval_generator, 输入pos samples, forward
    gen_loss = eval_generator(generator, eval_iter, nll_loss, args)
    topo_score = args.evaluation.topo_violation(samples)
    kl_divergence = args.evaluation.kl_divergence(samples)
    bleu_score = args.evaluation.bleu(samples)
    gen_pretrain_eval_loss.append(gen_loss)
    print("mse loss: {:.5f}, topo loss: {:.5}, kl divergence: {:.5}, bleu score: {:.5}\n".format(gen_loss, topo_score, kl_divergence, bleu_score))
    """
    #print('#####################################################\n\n')


# todo: rewrite d_l_train and sample , change the seq_len thing
def dloss_tloss_train(generator, optimizer):
    #  training
    """
    Train generator with the guidance of policy gradient
    """
    if ARGS.melting == 5:
        raise Exception("train_generator_DT should not be used")
    
    total_loss = 0.
    for _ in range(ARGS.dt_epochs):
        for i in range(100):
            # sample entrances
            seq_len = multinomial(ARGS.lens_dist, 1).item()
            if seq_len < 3:
                continue

            target_node_dist = ARGS.node_dist

            generator.ad_sample = True
            samples, node_dist, topo_errs = generator.sample(ARGS.batch_size, seq_len)
            generator.ad_sample = False

            if ARGS.melting in [0, 4]:
                loss = ARGS.beta * TOPO_LOSS(topo_errs, torch.zeros(topo_errs.size(), device=ARGS.device, dtype=torch.float)) + \
                    ARGS.gamma * D_LOSS(node_dist, target_node_dist)
            elif ARGS.melting == 1:
                loss =  ARGS.beta * TOPO_LOSS(topo_errs, torch.zeros(topo_errs.size(), device=ARGS.device, dtype=torch.float))
            elif ARGS.melting == 2:
                loss =  ARGS.gamma * D_LOSS(node_dist, target_node_dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / (ARGS.dt_epochs * 100)
    return avg_loss
    """
    samples = generate_samples(generator, args.batch_size, args.num_batches, NEGATIVE_FILE, args.lens_dist)
    gen_eval_iter = GenDataIter(POSITIVE_FILE, args.batch_size, args)
    gen_loss = eval_generator(generator, gen_eval_iter, nll_loss, args)
    topo_score = args.evaluation.topo_violation(samples)
    kl_divergence = args.evaluation.kl_divergence(samples)
    bleu_score = args.evaluation.bleu(samples)
    gen__eval_loss.append(gen_loss)
    print("gen mse loss: {:.5f}, topo score: {:.5f}, kl_divergence: {:.5}, bleu_score: {:.5}\n"
        .format(gen_loss, topo_score, kl_divergence, bleu_score))
    """
        
def train_(generator, dataset, optimizer, split_ratio):
    print('Start Pretraining')
    for i in range(math.floor(ARGS.pretrain_steps * split_ratio)):
        avg_loss = train_generator_MLE(dataset, generator, optimizer)
        print('pretrain epoch {}, loss: {:.5f}'.format(i, avg_loss))

    print('Start Training')
    for i in range(math.floor(ARGS.train_steps * split_ratio)):
        mle_avg_loss = train_generator_MLE(dataset, generator, optimizer)
        if ARGS.melting in [0, 1, 2]:
            dt_avg_loss = dloss_tloss_train(generator, optimizer)
            if i >= 14:
                generator.masked = True
        elif ARGS.melting == 3:
            mle_avg_loss = train_generator_MLE(dataset, generator, optimizer)
            if i >= 14 and ARGS.melting != 5:
                generator.masked = True
        elif ARGS.melting == 4:
            dt_avg_loss = dloss_tloss_train(generator, optimizer)
        print('train epoch {}, mle loss: {}, dt loss: {}'.format(i, mle_avg_loss, dt_avg_loss))

        """
        if i == ARGS.train_steps - 1:
            path = NEGATIVE_FILE + '-{}-{}-{}'.format(ARGS.melting, i)
            samples = generate_samples(generator, ARGS.dataset_size, path)
        if round % 7 == 0:
            samples = generate_samples(generator, args.batch_size, args.num_batches, NEGATIVE_FILE, args.lens_dist)
            bleu = args.evaluation.dataset_bleu(samples)
            topo_score = args.evaluation.topo_violation(samples)
            kl_divergence = args.evaluation.kl_divergence(samples)
            print('Bleu score for the generated dataset: {:.5}, topo: {:.5}, kl-divergence: {:.5}'.format(bleu, topo_score, kl_divergence))
        """

    # Save experiment data
    with open(ARGS.data_path + 'experiment.pkl', 'wb') as f:
        pkl.dump(
            (GEN_PRETRAIN_TRAIN_LOSS,
                GEN_PRETRAIN_EVAL_LOSS,
                GEN_EVAL_LOSS,
            ),
            f,
            protocol=pkl.HIGHEST_PROTOCOL
        )

def create_training_context(label=-1, labels=None, generator_path=None):
    # Create new model, dataset, and optimizer
    generator = None
    if generator_path:
        with open(generator_path, 'rb') as f:
            generator = pkl.load(f)
    else:
        generator = Generator(ARGS.vocab_size, ARGS.g_embed_dim, ARGS.g_hidden_dim, 
                          ARGS.cuda, ARGS.acc_mat, ARGS.g_state_embed_dim, ARGS.SOT)
    if ARGS.cuda:
        generator = generator.cuda()
    DATASET_FACTORY.set_labels(labels)
    dataset, node_dist, lens_dist = DATASET_FACTORY.build_dataset_for_cluster(label)
    optimizer = optim.Adam(params=generator.parameters(), lr=ARGS.gen_lr)
    ARGS.node_dist = node_dist
    ARGS.lens_dist = lens_dist
    return generator, dataset, optimizer

def save_model_as_path(generator, path):
    with open(path, 'wb') as f:
        pkl.dump(generator, f)

def train(n_cluster, split_ratio):
    ARGS.N_CLUSTERS = n_cluster
    ARGS.SPLIT_RATIO = split_ratio
    
    with open('./data/cluster_labels{}.pkl'.format(ARGS.N_CLUSTERS), 'rb') as f:
        labels = pkl.load(f)

    generator, dataset, optimizer = create_training_context(-1, labels)
    train_(generator, dataset, optimizer, ARGS.SPLIT_RATIO)
    save_model_as_path(generator, PARENT_GENERATOR_PATH)

    n_clusters = ARGS.N_CLUSTERS
    for i in range(n_clusters):
        label = i
        generator, dataset, optimizer = create_training_context(label, labels, PARENT_GENERATOR_PATH)
        train_(generator, dataset, optimizer, 1 - ARGS.SPLIT_RATIO)
        sub_generator_path = './data/sub_generator_cluster{}.pkl'.format(i)
        save_model_as_path(generator, sub_generator_path)
        SUB_GENERATORS_PATHS.append(sub_generator_path)

    negative_samples = {}
    for sub_generator_path in SUB_GENERATORS_PATHS:
        i = sub_generator_path.index('.pkl')
        label = int(sub_generator_path[i - 1])
        with open(sub_generator_path, 'rb') as f:
            sub_generator = pkl.load(f)
        n = np.count_nonzero(labels == label)
        path = './data/gene_cluster{}'.format(label)
        with torch.no_grad():
            samples = generate_samples(sub_generator, n, path)
        negative_samples[label] = samples

    with open('./data/negative_samples.pkl', 'wb') as f:
        pkl.dump(negative_samples, f)

    evaluator = Evaluator(POSITIVE_FILE, ARGS.interval, ARGS.acc_mat_numpy, ARGS.vocab_size, labels, ARGS.max_len, ARGS.N_CLUSTERS)
    metrics = ['node_dist', 'top100_dist', 'length_dist', 'topo_score', 'bleu_score', 'node_dist_cluster', 'top100_dist_cluster']
    result = evaluator.evaluate('./data/negative_samples.pkl', 'pkl', metrics)
    #print(result)

    with open('./data/result-{}-{}.pkl'.format(ARGS.N_CLUSTERS, ARGS.SPLIT_RATIO), 'wb') as f:
        pkl.dump(result, f)

if __name__ == '__main__':
    #n_clusters_lst = [2,3,4,5,6,7]
    """
    n_clusters_lst = [4,5,6,7]
    s_ratio_list = [0.7, 0.6, 0.5, 0.3, 0.2, 0.1]

    for n_cluster in n_clusters_lst:
        for s_ratio in s_ratio_list:
            try:
                train(n_cluster, s_ratio)
            except Exception as e:
                print(n_cluster, s_ratio, e)

    for n_cluster in n_clusters_lst:
        for s_ratio in s_ratio_list:
            with open('./data/result-{}-{}.pkl'.format(ARGS.N_CLUSTERS, ARGS.SPLIT_RATIO), 'rb') as f:
                result = pkl.load(f)
            print(result)
    """
    with open('./data/cluster_labels{}.pkl'.format(5), 'rb') as f:
        labels = pkl.load(f)
    
    ARGS.N_CLUSTERS = 5
    evaluator = Evaluator(POSITIVE_FILE, ARGS.interval, ARGS.acc_mat_numpy, ARGS.vocab_size, labels, ARGS.max_len, ARGS.N_CLUSTERS)
    result = evaluator.self_bleu()
    print(result)