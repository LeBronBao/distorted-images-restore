from tqdm import tqdm
from models import CompletionNetwork, Discriminator
from datasets import ImageDataset
from losses import completion_network_loss
from utils import (
    process_image,
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    poisson_blend,
)
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import os
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default="datasets/clipped_image")
parser.add_argument('--result_dir',type=str,default="./results/")
parser.add_argument('--step', type=int, default=50000)
parser.add_argument('--snaperiod', type=int, default=5000)  # 每多少步保存一次
parser.add_argument('--cn_input_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam'], default='adadelta')
parser.add_argument('--bsize', type=int, default=16)
parser.add_argument('--num_gpus', type=int, choices=[1, 2], default=1)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--resume', type=str, default="No")
parser.add_argument('--model_cn', type=str, default='')  # 保存生成器模型的名字
parser.add_argument('--model_cd', type=str, default='')  # 保存判别器模型的名字
parser.add_argument('--train_step', type=int, default=0)  # 已经训练的步数，与上面两步一起保存


def main(args):

  # ================================================
    # Preparation
    # ================================================
    # args.data_dir = os.path.expanduser(args.data_dir)
    # args.result_dir = os.path.expanduser(args.result_dir)

    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')

    if args.num_gpus == 1:
        # train models in a single gpu
        gpu_cn = torch.device('cuda:0')
        gpu_cd = gpu_cn
    else:
        # train models in different two gpus
        gpu_cn = torch.device('cuda:0')
        gpu_cd = torch.device('cuda:1')

    # create result directory (if necessary)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for s in ['result']:
        if not os.path.exists(os.path.join(args.result_dir, s)):
            os.makedirs(os.path.join(args.result_dir, s))

    # dataset
    trnsfm = transforms.Compose([
        transforms.Resize(args.cn_input_size),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset... (it may take a few minutes)')
    train_dset = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm)
    test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm)
    train_loader = DataLoader(train_dset, batch_size=args.bsize, shuffle=True)

    if args.resume == "No": 
        # save training config
        args_dict = vars(args)
        with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
            json.dump(args_dict, f)  # 将json信息写进文件

        # 初始化生成器和判别器
        model_cn = CompletionNetwork()
        model_cn = model_cn.to(gpu_cn)

        model_cd = Discriminator(input_shape=(3, args.cn_input_size, args.cn_input_size))
        model_cd = model_cd.to(gpu_cd)

    else:
        with open("./results/config.json", 'r') as f:
            config = json.load(f)  # 加载json信息
        args.train_step = config['train_step']
        args.model_cn = config["model_cn"]
        args.model_cd = config["model_cd"]

        model_cn = CompletionNetwork()
        model_cn.load_state_dict(torch.load(args.model_cn, map_location='cpu')) 
        model_cn = model_cn.to(gpu_cn)

        model_cd = Discriminator(input_shape=(3, args.cn_input_size, args.cn_input_size))
        model_cd.load_state_dict(torch.load(args.model_cd, map_location='cpu'))
        model_cd = model_cd.to(gpu_cd)

    # 选择优化方法
    if args.optimizer == 'adadelta':
        opt_cn = Adadelta(model_cn.parameters())
    else:
        opt_cn = Adam(model_cn.parameters())
    if args.optimizer == 'adadelta':
        opt_cd = Adadelta(model_cd.parameters())
    else:
        opt_cd = Adam(model_cd.parameters())

    # 损失函数，二元交叉熵
    criterion_cd = BCELoss()

    alpha = torch.tensor(args.alpha).to(gpu_cd)
    pbar = tqdm(total=(500000 - args.train_step))  # 进度条
    while pbar.n < (500000 - args.train_step):
        for x in train_loader:
            x_rotate = process_image(x)  # 旋转图片
            x_rotate = x_rotate.to(gpu_cn)
            x = x.to(gpu_cn)

            # ================================================
            # train model_cd  训练判别器
            # ================================================
            opt_cd.zero_grad()  # 清空梯度

            # fake
            fake = torch.zeros((len(x), 1)).to(gpu_cd)  # 作为负类标签

            output_cn = model_cn(x_rotate)
            input_fake = output_cn.detach()
            input_fake = input_fake.to(gpu_cd)
            output_fake = model_cd(input_fake)
            loss_cd_1 = criterion_cd(output_fake, fake)  # 判别器的第一个loss，计算生成图片和负类标签的loss

            # real
            real = torch.ones((len(x), 1)).to(gpu_cd)  # 作为正类标签
            input_real = x
            input_real = input_real.to(gpu_cd)
            output_real = model_cd(input_real)
            loss_cd_2 = criterion_cd(output_real, real)  # 判别器的第二个loss，计算真实图片和正类标签的loss

            # optimize
            loss_cd = (loss_cd_1 + loss_cd_2) * alpha / 2.
            loss_cd.backward()
            opt_cd.step()

            # ================================================
            # train model_cn  训练生成器
            # ================================================
            opt_cn.zero_grad()  # 清空梯度

            loss_cn_1 = completion_network_loss(x, output_cn).to(gpu_cd)
            input_gd_fake = output_cn
            input_fake = input_gd_fake.to(gpu_cd)
            output_fake = model_cd(input_fake)
            loss_cn_2 = criterion_cd(output_fake, real)  # 生成器loss，计算真实图片和生成图片的loss

            # optimize
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.
            loss_cn.backward()
            opt_cn.step()

            msg = 'train |'
            msg += ' train loss (cd): %.5f' % loss_cd.cpu()
            msg += ' train loss (cn): %.5f' % loss_cn.cpu()
            pbar.set_description(msg)
            pbar.update()

            # test
            if pbar.n % args.snaperiod == 0:
                args_dict = vars(args)
                with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
                    args_dict['train_step'] += args.snaperiod 
                    args_dict['resume'] = "Yes" 
                    args_dict['model_cn'] = "./results/result/" + 'model_cn_step%d' % args_dict['train_step']
                    args_dict['model_cd'] = "./results/result/" + 'model_cd_step%d' % args_dict['train_step']
                    json.dump(args_dict, f)
                
                with torch.no_grad():
                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x_rotate = process_image(x)
                    #x_rotate = x_rotate.to(gpu_cn)
                    #x = x.to(gpu_cn)
                    output = model_cn(x_rotate)
                    imgs = torch.cat((x_rotate.cpu(), output.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'result', 'step%d.png' % args_dict['train_step']), nrow=len(x))
                    torch.save(model_cn.state_dict(), os.path.join(args.result_dir, 'result', 'model_cn_step%d' % args_dict['train_step']))
                    torch.save(model_cd.state_dict(), os.path.join(args.result_dir, 'result', 'model_cd_step%d' % args_dict['train_step']))

            if pbar.n >= args.step:
                break
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
