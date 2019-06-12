from tqdm import tqdm
from models import CompletionNetwork, Discriminator
from datasets import ImageDataset, ImagePairsDataset
from losses import completion_network_loss
from utils import (
    process_image,
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
    sample_random_test_batch,
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
parser.add_argument('--data_dir', type=str, default="datasets/distorted_images")
parser.add_argument('--result_dir', type=str, default="./results/")
parser.add_argument('--step', type=int, default=500000)
parser.add_argument('--snaperiod', type=int, default=5000)  # 每多少步保存一次
parser.add_argument('--cn_input_height', type=int, default=256)  # 输入输出图片尺寸
parser.add_argument('--cn_input_width', type=int, default=256)
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


    if args.resume == "No":
        # save training config
        args_dict = vars(args)
        with open(os.path.join(args.result_dir, 'config.json'), mode='w') as f:
            json.dump(args_dict, f)  # 将json信息写进文件

        # 初始化生成器和判别器
        model_cn = CompletionNetwork()
        model_cn = model_cn.to(gpu_cn)

        model_cd = Discriminator(input_shape=(3, args.cn_input_height, args.cn_input_width))
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

        model_cd = Discriminator(input_shape=(3, args.cn_input_height, args.cn_input_width))
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

    # dataset
    image_pair_dataset = ImagePairsDataset(args.data_dir, args.bsize, 0.8, args.cn_input_height, args.cn_input_width)
    train_loader, test_loader = image_pair_dataset.get_train_test_tensor()

    alpha = torch.tensor(args.alpha).to(gpu_cd)
    pbar = tqdm(total=(500000 - args.train_step))  # 进度条
    while pbar.n < (500000 - args.train_step):
        for batch in train_loader:
            ori_x = batch[0]  # 原图
            distort_x = batch[1]  # 扭曲后的图片

            ori_x = ori_x.to(gpu_cn)
            distort_x = distort_x.to(gpu_cn)

            # ================================================
            # train model_cd  训练判别器
            # ================================================
            opt_cd.zero_grad()  # 清空梯度

            # fake
            fake = torch.zeros((len(ori_x), 1)).to(gpu_cd)  # 作为负类标签

            output_cn = model_cn(distort_x)  # 生成器根据扭曲图片生成一张图片
            input_fake = output_cn.detach()
            input_fake = input_fake.to(gpu_cd)
            output_fake = model_cd(input_fake)  # 判别器判断生成图片的标签
            loss_cd_1 = criterion_cd(output_fake, fake)  # 判别器的第一个loss，计算生成图片标签和负类标签的loss

            # real
            real = torch.ones((len(ori_x), 1)).to(gpu_cd)  # 作为正类标签
            input_real = ori_x
            input_real = input_real.to(gpu_cd)
            output_real = model_cd(input_real)  # 判别器判断真实图片的标签
            loss_cd_2 = criterion_cd(output_real, real)  # 判别器的第二个loss，计算真实图片标签和正类标签的loss

            # optimize
            loss_cd = (loss_cd_1 + loss_cd_2) * alpha / 2.
            loss_cd.backward()
            opt_cd.step()

            # ================================================
            # train model_cn  训练生成器
            # ================================================
            opt_cn.zero_grad()  # 清空梯度

            loss_cn_1 = completion_network_loss(ori_x, output_cn).to(gpu_cd)  # 真实图片和生成图片的MSE loss
            input_gd_fake = output_cn
            input_fake = input_gd_fake.to(gpu_cd)
            output_fake = model_cd(input_fake)
            loss_cn_2 = criterion_cd(output_fake, real)  # 生成器loss，计算真实图片和生成图片的BCE loss

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
                    distort_x = sample_random_test_batch(test_loader)
                    distort_x = distort_x.to(gpu_cn)
                    output = model_cn(distort_x)
                    imgs = torch.cat((distort_x.cpu(), output.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'result', 'step%d.png' % args_dict['train_step']),
                               nrow=len(distort_x))
                    torch.save(model_cn.state_dict(),
                               os.path.join(args.result_dir, 'result', 'model_cn_step%d' % args_dict['train_step']))
                    torch.save(model_cd.state_dict(),
                               os.path.join(args.result_dir, 'result', 'model_cd_step%d' % args_dict['train_step']))

            if pbar.n >= args.step:
                break
    pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)