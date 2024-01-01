import sys
import os.path
import numpy as np
import torch
import datetime

sys.path.append(os.path.split(sys.path[0])[0])


import math
from generator import Generator
from discriminator import Discriminator
from loss import PGLoss
from rollout import Rollout
import pickle as pkl
from epochBuilder import EpochBuilder
from data_iter import GenDataIter

class Env :
    def __init__(self) -> None:
        pass

env = Env()

device = torch.device('cuda:0')
env.device = device
print('device: ' + torch.cuda.get_device_name(0))
env.num_nodes = 2751
env.num_batch = 100
env.output_file = "./baselines/SeqGAN/data/gene.data"

# todo: loss/reward正负号/优化目标是最大化还是最小化
with open("./baselines/SeqGAN/data/entrance_distribution.pkl", "rb") as f:
    env.entrance_distribution = pkl.load(f, encoding="iso-8859-1").to(device).to(torch.float)

with open("./baselines/SeqGAN/data/one_step_accessiblity_matrix_.pkl", "rb") as f:
    env.connectivity = pkl.load(f, encoding="iso-8859-1").to(device)

with open("./baselines/SeqGAN/data/stay_distribution.pkl", "rb") as f:
    tmp = torch.ones(env.num_nodes, 3000, device=env.device)
    attraction = pkl.load(f, encoding="iso-8859-1").to(device)
    tmp[:, 0:864] = attraction
    env.attraction = tmp

with open("./baselines/SeqGAN/data/idx_s_t_tensor.pkl", "rb") as f:
    env.idx_s_t_tensor = pkl.load(f, encoding="iso-8859-1").to(device)

with open("./baselines/SeqGAN/data/start_time_and_traj_lens_joint_distribution.pkl", "rb") as f:
    env.startTime_trajLens_joint_dist = pkl.load(f, encoding="iso-8859-1").to(device).to(torch.float)

with open("./baselines/SeqGAN/data/lens_distribution.pkl", "rb") as f:
    env.lens_dist = pkl.load(f, encoding="iso-8859-1").to(device).to(torch.float)

with open("./baselines/SeqGAN/data/day_1_2_dataset.pkl", "rb") as f:
    train_dataset = pkl.load(f, encoding="iso-8859-1")["train"]

for len in train_dataset:
    train_dataset[len] = train_dataset[len].to(device).to(torch.long)

env.seed_generator = torch.Generator(device=device)
env.seed_generator.manual_seed(3)
env.entrance_idx = torch.as_tensor([16, 0, 24, 47], device=env.device, dtype=torch.int64)

# args for generator
env.node_embedding_dim = 8
env.node_embeddings = torch.FloatTensor(np.random.rand(env.num_nodes + 1, env.node_embedding_dim)).to(env.device).requires_grad_()
env.dur_embedding_dim = 8
env.num_durs = 180
env.dur_embeddings = torch.FloatTensor(np.random.rand(env.num_durs, env.dur_embedding_dim)).to(env.device).requires_grad_()
env.gen_hidden_dim = 32
env.activation = "leaky_relu"
env.rescale_rewards = False
env.sigma = 12.0
env.gen_random_size = 2
env.num_poi = 25
env.gen_expert_dim = 8
env.is_rolling_out = False

env.filter_sizes = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64, device=env.device)
env.num_filters = torch.tensor([20, 20, 20, 20, 10, 10], dtype=torch.int64, device=env.device)
env.dis_dropout_prob = 0.2

env.disc_gru_hidden_dim = 10
env.disc_gru_dropout = 0.2
env.batch_size = 64
env.gen_ad_epoch = 0
env.trainDis_epoch = 0

env.trainGen_batch_per_epoch = 32
env.rollout_num = 4

env.training_gen = False

def multinomial(dist, num):
    return torch.multinomial(dist, num, replacement=True, generator=env.seed_generator).squeeze()

def sample_negatives(generator:Generator, is_training_gen):
    # sample one batch of negatives
    env.training_gen = is_training_gen

    joint_dist = env.startTime_trajLens_joint_dist
    lens_dist = env.lens_dist
    entrance_distribution =  env.entrance_distribution
    start_nodes = []

    seq_len = multinomial(lens_dist, 1).item()
    start_time_dist = joint_dist[:, seq_len]
    traj_num = env.batch_size

    start_time = multinomial(start_time_dist, traj_num)
    for i in range(traj_num):
        entrance = multinomial(entrance_distribution[:, start_time[i]], 1).squeeze()
        entrance = env.entrance_idx[entrance]
        start_nodes.append(entrance.unsqueeze(dim=0))

    start_nodes = torch.stack(start_nodes, dim=0)
    start_node_dist = torch.ones(traj_num, env.num_nodes, dtype=torch.float, device=env.device) * 0.00036

    rand_var = torch.rand(traj_num, env.gen_random_size, device=env.device)
    neg_samples, log_node_probs = generator.sample(seq_len, start_nodes, rand_var, start_time, start_node_dist)
    env.training_gen = False

    return neg_samples, log_node_probs, rand_var, start_time, start_node_dist, seq_len

def generate_samples(generator:Generator, is_training_gen, num_batches, output_file):

    samples_dict = {}
    for i in range(num_batches):
        samples, _, _, _, _, seq_len = sample_negatives(generator, is_training_gen)
        if seq_len not in samples_dict:
            samples_dict[seq_len] = samples.to('cpu').numpy().tolist()
        else:
            samples_dict[seq_len] += samples.to('cpu').numpy().tolist()
    with open(output_file, 'w') as fout:
        for seq_len in samples_dict:
            sample = samples_dict[seq_len]
            fout.write('{}\n'.format(str(seq_len)))
            for seq in sample:
                string = ' '.join([str(s) for s in seq])
                fout.write('{}\n'.format(string))

def train_generator_MLE(gen, criterion, optimizer, epochs, 
        gen_pretrain_train_loss, args):
    """
    Train generator with MLE
    """
    avg_loss = None
    gen_data_iter = GenDataIter("./baselines/SeqGAN/real.data", args.batch_size, args)
    for epoch in range(epochs):
        total_loss = 0.
        for data, target in gen_data_iter:
            data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            output = gen(data)
            loss = criterion(output, target)
            total_loss += loss.detach().to('cpu').numpy().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        gen_data_iter.reset()
        avg_loss = total_loss / gen_data_iter.num_batches

    print("Epoch {}, train loss: {:.5f}".format(epoch, avg_loss))
    gen_pretrain_train_loss.append(avg_loss)

def train_generator_PG(generator:Generator, discriminator:Discriminator, rollout:Rollout, pgloss:PGLoss, gen_optimizer):
    """
    Policy Gradient 训练生成器
    """
    total_loss = 0
    num_batch = env.num_batch
    for _ in range(num_batch):
        # 根据起始点概率矩阵采样起始点
        neg_samples, log_node_probs, rand_var, start_time, start_node_dist, seq_len = sample_negatives(generator, True) # 一次只生成一样长度的轨迹
        if seq_len < 3:
            continue
        now = datetime.datetime.now()
        # print("\nCurrent batch seq_len: " + str(seq_len) + "; " + "started at: " + now.strftime("%Y-%m-%d %H:%M:%S"))
        # 计算奖励
        reward = rollout.get_reward(neg_samples, discriminator, rand_var, start_time, start_node_dist)
        # 计算损失
        loss = pgloss(log_node_probs, neg_samples, reward, seq_len)

        # 反向传播
        gen_optimizer.zero_grad()
        
        loss.backward()
        tmp = 0
        for para in generator.parameters():
            if torch.isnan(torch.sum(para.grad)):
                print("is_nan")
            elif _ > 0:
                tmp += torch.sum(torch.abs(para.grad))

        torch.nn.utils.clip_grad_value_(generator.parameters(), 0.05)
        gen_optimizer.step()
        # 更新rollout实例的生成器
        rollout.update_params()

def get_correct_count(pred, target):
    """
    Input:\n
        -pred: 判别器给出的真伪概率; tensor: (num_traj * 2), 第一列为伪的概率, 第二列为真的概率;\n
        -target: 真伪label; tensor: (num_traj); 真为1, 伪为0;\n
    Output:\n
        -correct_count: 正确率; float\n
    """
    predicted_true = pred[:, 1] >= math.log(0.5)
    predicted_false = pred[:, 0] >= math.log(0.5)
    true_is_true = predicted_true * target
    false_is_false = predicted_false * (1 - target)
    true_count = torch.sum(true_is_true) + torch.sum(false_is_false)
    return true_count

def train_discriminator(discriminator:Discriminator, generator:Generator, dis_optimizer, nll_loss,
        dis_adversarial_train_loss, dis_adversarial_train_acc, train_dataset):
    """
    Train discriminator
    """   
    batch_builder = EpochBuilder(env, generator, train_dataset)
    correct = 0
    total_loss = 0.
    count = 0
    for seqs, labels in batch_builder:
        pred = discriminator(seqs)
        correct += get_correct_count(pred, labels)
        loss = nll_loss(pred, labels)
        total_loss += loss.item()
        dis_optimizer.zero_grad()
        loss.backward()
        dis_optimizer.step()
        count += 1

    avg_loss = total_loss / batch_builder.num_batch
    acc = correct.item() * 1.00 / (env.num_batch * env.batch_size)
    print("Epoch {}, train loss: {:.5f}, train acc: {:.3f}".format(env.trainDis_epoch, avg_loss, acc))
    env.trainDis_epoch += 1
    dis_adversarial_train_loss.append(avg_loss)
    dis_adversarial_train_acc.append(acc)
    return acc

def eval_generator(discriminator:Discriminator, generator:Generator):
    accuracy = 0
    preds = []
    for i in range(40):
        neg_samples, _, _, _, _, _ = sample_negatives(generator, False)
        pred = discriminator(neg_samples)
        preds.append(pred)
    preds = torch.cat(preds, dim=0)
    label = torch.zeros(40 * env.batch_size, device=env.device)
    accuracy = torch.div(get_correct_count(preds, label), 40.0 * env.batch_size)
    return preds, accuracy

def adversarial_train_(gen, dis, rollout, pg_loss, nll_loss, gen_optimizer, dis_optimizer, 
        dis_adversarial_train_loss, dis_adversarial_train_acc, train_dataset):
    """
    Adversarially train generator and discriminator
    """
    acc = 0
    count = 0
    while count < 3:
        now = datetime.datetime.now()
        #print("\nTrain discriminator started at " + now.strftime("%Y-%m-%d %H:%M:%S"))
        acc = train_discriminator(dis, gen, dis_optimizer, nll_loss, dis_adversarial_train_loss, dis_adversarial_train_acc, train_dataset)
        print("\nDiscriminator accuracy: " + str(acc))
        count += 1
        now = datetime.datetime.now()
        #print("\nTrain discriminator ended at " + now.strftime("%Y-%m-%d %H:%M:%S"))

    count = 0
    while count < 2:
        now = datetime.datetime.now()
        print("\nTrain generator started at " + now.strftime("%Y-%m-%d %H:%M:%S"))
        train_generator_PG(gen, dis, rollout, pg_loss, gen_optimizer)
        pred, accuracy = eval_generator(dis, gen)
        avg_pred = torch.mean(pred[:, 1])
        now = datetime.datetime.now()
        print("Fool rate:{}, average prediction: {}".format(1- accuracy, avg_pred))
        if accuracy < 0.1: break
        count += 1

    rollout.update_params()
    generate_samples(generator, False, 80, env.output_file)

generator = Generator(env).to(device)
node_prefix = torch.tensor(np.ones((64, 1)) * 200, dtype=torch.int64, device=env.device)
dur_prefix = torch.tensor(np.random.rand(64, 1), device=env.device)

discriminator = Discriminator(env).to(env.device)
rollout = Rollout(generator, env)
pgloss = PGLoss(env)

gen_optim = torch.optim.Adam(generator.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08)
dis_optim = torch.optim.SGD(discriminator.parameters(), lr=0.005, weight_decay=0.0001)
dis_adversarial_train_loss = []
dis_adversarial_train_acc = []

#train_generator_MLE(generator, torch.nn.NLLLoss(), gen_optim, 10, [], env)
generate_samples(generator, False, 80, env.output_file)
for i in range(50):
    now = datetime.datetime.now()
    print("\nRound : " + str(i) + " " + "started at: " + now.strftime("%Y-%m-%d %H:%M:%S"))
    adversarial_train_(generator, discriminator, rollout, pgloss, torch.nn.NLLLoss(), gen_optim, dis_optim, dis_adversarial_train_loss, dis_adversarial_train_acc, train_dataset)