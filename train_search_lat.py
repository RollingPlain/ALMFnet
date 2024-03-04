import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from model_search import Network_DAG_lat as Network
from architect import Architect
from dataset import TrainingData
from pathlib import Path
from tqdm import tqdm
from kornia.utils import tensor_to_image
from torch.utils.data import DataLoader
import pickle


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--dataroot', type=str, default='data', help='location of the data corpus')
parser.add_argument('--phase', type=str, default='train', help='location of the data corpus')
parser.add_argument('--id', type=str, default='ALMFNet', help='location of the data corpus')
parser.add_argument('--lat', type=str, default='./latency_gpu.pkl', help='location of the latency pkl')
parser.add_argument('--length', type=int, default=100, help='length of the dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--img_size', type=int, default=128, help='image size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.75, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=4e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
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
    
    lat_pth = args.lat
    with open(lat_pth, 'rb') as f:
        lat_lookup = pickle.load(f)

    model = Network(args.init_channels, l1_loss, mse_loss, lat_lookup)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)


    dataset = TrainingData(dataroot=args.dataroot,phase=args.phase,finesize=args.img_size)
    
    cache = Path('cache_search')
    name = args.id
    rst = cache / name
    rst.mkdir(parents=True, exist_ok=True)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=8)

    valid_queue = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(1, args.epochs + 1):

        lr = scheduler.get_last_lr()

        print('epoch %d lr %e', epoch, lr[0])

        # training
        train(train_queue, valid_queue, model, architect, optimizer, lr, epoch, rst)

        genotype = model.genotype()

        if epoch == args.epochs:
             torch.save(model, rst / f'final.pt')
        scheduler.step()


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch, rst):
    model.train()

    fus = None
    loss_rec = []

    process = tqdm(enumerate(train_queue), total=len(train_queue))

    for batch_idx, (set_ct) in process:

        if torch.cuda.is_available():
            set_ct = set_ct.cuda()

        # get a random minibatch from the search queue with replacement
        if epoch > 0 and batch_idx %1 == 0:

            val_set_ct = next(iter(valid_queue))

            if torch.cuda.is_available():
                val_set_ct = val_set_ct.cuda()

            architect.step(val_set_ct, lr, optimizer, lat = True)

        if epoch > 0 and batch_idx %4 == 0:
            l1=[]
            w_list = model.arch_parameters()
            w_f1 = F.softmax(w_list[0], dim=-1).data.cpu().numpy()
            for i in range(len(w_f1)):
                l1.append(max(w_f1[i]))

        list = [set_ct]
        for i in range(0, len(list)):
            mri, oth, mri_mask, oth_mask, seg = torch.chunk(list[i], 5, dim=1)
            fus, lat = model(mri, oth)

            loss = model._loss(fus, mri, oth, mri_mask, oth_mask, seg)
            if epoch < 2:
                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            loss_rec.append(loss.item())
            process.set_description(f'epoch: {epoch} | loss: {np.mean(loss_rec):.03f}({loss.item():.03f})')

        if epoch > 1 and batch_idx % 10 == 0:
            name = str(epoch).zfill(2) + '_' + str(batch_idx).zfill(2)
            torch.save(model, rst / f'{name}.pt')
        


if __name__ == '__main__':
    main()
