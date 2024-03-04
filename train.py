import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from model_DAG import Network_DAG as Network
import genotypes
from dataset import TrainingData
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from loss import FusionLoss

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataroot', type=str, default='data', help='location of the data corpus')
parser.add_argument('--phase', type=str, default='train', help='location of the data corpus')
parser.add_argument('--id', type=str, default='ALMFNet', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=128, help='image size')
parser.add_argument('--arch', type=str, default='DAG_lat', help='which architecture to use')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()


def main():
    if not torch.cuda.is_available():
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    if torch.cuda.is_available():
        l1_loss = l1_loss.cuda()
        mse_loss = mse_loss.cuda()

    genotype = eval("genotypes.%s" % args.arch)

    model = Network(args.init_channels, genotype)
    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate)

    dataset = TrainingData(dataroot=args.dataroot,phase=args.phase,finesize=args.img_size)
    train_queue = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    cache = Path('cache_train')
    name = args.id
    rst = cache / name
    rst.mkdir(parents=True, exist_ok=True)


    for epoch in range(1, args.epochs + 1):


        train(train_queue, model, optimizer, epoch, l1_loss, mse_loss)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), rst / f'{str(epoch).zfill(3)}.pt')


def train(train_queue, model, optimizer, epoch, l1_loss, mse_loss):
    model.train()

    fus = None
    loss_rec = []

    process = tqdm(enumerate(train_queue), total=len(train_queue))

    for batch_idx, (set_ct) in process:

        if torch.cuda.is_available():
            set_ct = set_ct.cuda()
    
        list = [set_ct]
        for i in range(0, len(list)):

            mri, oth, mri_mask, oth_mask, seg = torch.chunk(list[i], 5, dim=1)
            fus = model(mri, oth)

            optimizer.zero_grad()

            loss = FusionLoss(fus, mri, oth, mri_mask, oth_mask, seg, l1_loss, mse_loss)
            loss.backward()
            optimizer.step()

            loss_rec.append(loss.item())
            process.set_description(f'epoch: {epoch} | loss: {np.mean(loss_rec):.03f}({loss.item():.03f})')

    t_fus = fus[0].cpu().detach()


if __name__ == '__main__':
    main()
