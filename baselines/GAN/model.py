import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
h_dim = 400
batch_size = 64
img_size = 32
nc = 16
nz = 16
ngf = 16
ndf = 16
test_count = 10928
data_path = './baselines/GAN/data/data'
n_epoch = 182
disc_step = 10
gp_coefficient = 10
result_size = 1000

class TrajDataset(Dataset):
    def __init__(self, data_path, data_count):
        self.data_count = data_count
        self.data_path = data_path

    def __len__(self):
        return self.data_count

    def __getitem__(self, idx):
        img_path = self.data_path + '{}.pkl'.format(idx)
        with open(img_path, 'rb') as f:
            traj = pkl.load(f)
        return traj, True

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 32 x 32
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, 1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 1 x 4 x 4
        )
        self.output = nn.Linear(16, 1)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 16)
        out = self.output(out)
        return out.view(-1)

# 生成图像
def generate_image(D, G, x_r, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda()  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()

    # draw samples
    with torch.no_grad():
        z = torch.randn(batch_size, 2).cuda()  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
    plt.scatter(x_r[:, 0], x_r[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

# NN的参数模型初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def gradient_penalty(D, x_r, x_f):
    '''
    :param D:
    :param x_r: [b, 2]
    :param x_f: [b, 2]
    :return:
    '''
    # 根据x的线性插值的公式
    # 维度: [b,1]
    t = torch.rand(x_f.size(dim=0), nc, img_size, img_size).to(device)  #t[0,1]
    # interpolation(插值)
    mid = t*x_r+(1-t)*x_f
    # 设置mid可以求导  （因为公式中GP部分需要导数）
    mid.requires_grad_()
    # 得到输出
    pred = D(mid)
    # 求导数
    grads = torch.autograd.grad(outputs=pred,
                        inputs=mid,
                        grad_outputs=torch.ones_like(pred),
                        create_graph=True,#这是为了二阶求导
                        retain_graph=True,#如果需要backward则需要保留梯度信息
                        only_inputs=True
                        )[0]
    # grads.norm(2,dim=1)  
    # 2范数
    # 根据Loss的计算公式可知  
    # 2范数与1越接近越好
    gp = torch.pow(grads.norm(2, dim=1)-1,2).mean()
    return gp

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().to(device)
    D = Discriminator().to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    # 设置G与D优化器
    optim_G = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))

    # GAN的核心部分
    print("statrt training")
    for epoch in range(n_epoch):
        st_time = time.time()
        data_iter = iter(DataLoader(TrajDataset(data_path, test_count), batch_size=batch_size, shuffle=True, num_workers=4))
        
        gp = 0

        for batch in range(math.floor(test_count * 1.0 / (batch_size * disc_step))):
            # 1. Train Discriminator Firstly
            for _ in range(disc_step):  # 暂定优化五步
                # 训练real_data
                x_r = next(data_iter)[0] #一组为batch_size大小
                # 数据处理: 把数据设置为tensor
                x_r = x_r.to(device)
                # [d, 2]-->[d, 1]
                pred_r = (D(x_r))
                # 最大化pred_r, 最小化loss_r(所以加上符号)
                loss_r = -pred_r.mean()

                # 训练fake_data
                # 创建假数据
                z = torch.randn(x_r.size(dim=0), nz, 1, 1).to(device)
                x_f = G(z).detach()  # 因为不优化G，detach()梯度计算停止

                # 判别假数据
                pred_f = (D(x_f))
                loss_f = (pred_f.mean())

                # 设置GP
                gp = gradient_penalty(D,x_r,x_f)

                # 合并损失
                loss_D = loss_r + loss_f + gp_coefficient * gp

                # 优化
                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            # 2. Train Generator
            # 创建 fake 数据 --> 初始给 G 的噪声一样
            z2 = torch.randn(batch_size, nz, 1, 1).to(device)
            # 使用 G 生
            x_f2 = G(z2)
            pred_f2 = (D(x_f2))  # 这里虽然不优化 D 但是也没有用 detach 因为他在 G 的后面
            # 这是优化 G, 所以需要 pred_f2 越大越好，loss_G 越小越好（故取负数）
            loss_G = -pred_f2.mean()
            # 优化
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
        elapsed = time.time() - st_time

        if epoch % 5 == 0:
            print('Epoch {:.5f}: loss_d {:.5f}, loss_g {:.5f}; {:.3f} seconds elapsed when processing this epoch'
                .format(epoch, -(loss_D - gp_coefficient * gp).item(), loss_G.item(), elapsed))

        if epoch == 180:
            z = torch.randn(result_size, nz, 1, 1).to(device)
            x = G(z)
            with open("./baselines/GAN/results/trajs_epoch{}".format(epoch), "wb") as f:
                pkl.dump(x, f)
            torch.save(G.state_dict(), './baselines/GAN/params/G_param_epoch{}'.format(epoch))
            torch.save(D.state_dict(), './baselines/GAN/params/D_param_epoch{}'.format(epoch))
            print("Epoch {} saving finished".format(epoch))

if __name__ == "__main__":
    main()