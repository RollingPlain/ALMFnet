import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from operations import ReLUConvBN
from torch.autograd import Variable
from genotypes import PRIMITIVES
from collections import namedtuple
from tools import Sobelxy


class MixedOp(nn.Module):

    def __init__(self, C):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C)
            self._ops.append(op)

    def forward(self, x, weights):

        return sum(w * op(x) for w, op in zip(weights, self._ops))


class CellMIF(nn.Module):

    def __init__(self, C, steps=4, multiplier=4):
        super(CellMIF, self).__init__()

        self.preprocess0 = ReLUConvBN(C, C, 1, 1, 0, affine=False)
        self._steps = steps 
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1 + i):
                stride = 1
                op = MixedOp(C)
                self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess0(s0)

        states = [s0]
        offset = 0
        for i in range(self._steps): 
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

class CellMIF_lat(nn.Module):

    def __init__(self, C, latency, steps=4, multiplier=4): 
        super(CellMIF_lat, self).__init__()

        self.preprocess0 = ReLUConvBN(C, C, 1, 1, 0, affine=False)
        self._steps = steps  
        self._multiplier = multiplier
        self.latency = latency

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(1 + i):
                stride = 1
                op = MixedOp(C)
                self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess0(s0)

        states = [s0]
        offset = 0
        for i in range(self._steps):  
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        lat = 0 
        for i in range(weights.shape[0]): 
            lat += sum(self.latency[k] * weights[i][j] for j, k in enumerate(PRIMITIVES))
        

        return torch.cat(states[-self._multiplier:], dim=1), lat


class Network_DAG(nn.Module):

    def __init__(self, C, criterion1, criterion2, steps=4, multiplier=4):
        super(Network_DAG, self).__init__()
        self._C = C
        self.l1_loss = criterion1
        self.mse_loss = criterion2
        self._steps = steps
        self._multiplier = multiplier

        self.stem1 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(C*8, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(C*4, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, 1, 3, padding=1, bias=False),
            nn.Tanh()
        )


        self.cell_f1 = CellMIF(C, steps=self._steps, multiplier=self._multiplier)
        self.cell_f2 = CellMIF(C, steps=self._steps, multiplier=self._multiplier)
        self.cell_f = CellMIF(C, steps=self._steps, multiplier=self._multiplier)

        self._initialize_alphas()

    def new(self):
        model_new = Network_DAG(self._C, self.l1_loss, self.mse_loss).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x1, x2):
        f1 = self.stem1(x1)
        f2 = self.stem2(x2)

        weights_f1 = F.softmax(self.alphas_f1, dim=-1)
        weights_f2 = F.softmax(self.alphas_f2, dim=-1)

        f1_out = self.cell_f1(f1, weights_f1)
        f2_out = self.cell_f2(f2, weights_f2)

        f_in = torch.cat([f1_out, f2_out], dim=1)

        weights_f = F.softmax(self.alphas_f, dim=-1)
        f_in1 = self.fusion1(f_in)

        f_out = self.cell_f(f_in1, weights_f)
        fus = self.fusion2(f_out)

        return fus

    def _loss(self, fus, mri, oth, mri_mask, oth_mask, seg):

        # Pixel Loss
        loss_pixel = self.mse_loss(fus * mri_mask, mri * mri_mask) + self.mse_loss(fus * oth_mask, oth * oth_mask)

        # Gradient Loss
        sobelconv = Sobelxy()
        grad_max = torch.max(sobelconv(mri),sobelconv(oth))
        grad_fus = sobelconv(fus)
        loss_grad = self.l1_loss(grad_fus, grad_max)

        # Segment Loss
        seg = torch.where(seg <= 0, torch.ones_like(seg) * 0.25, torch.ones_like(seg) * 0.75)
        loss_seg = self.l1_loss(fus * seg, mri * seg)
        
        # Total Loss
        loss_total =  loss_pixel + 0.5* loss_grad + 0.05* loss_seg

        return loss_total

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i)) #node之间总共几根线
        num_ops = len(PRIMITIVES)

        self.alphas_f1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_f2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_f = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_f1,
            self.alphas_f2,
            self.alphas_f
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        gene_f1 = _parse(F.softmax(self.alphas_f1, dim=-1).data.cpu().numpy())
        gene_f2 = _parse(F.softmax(self.alphas_f2, dim=-1).data.cpu().numpy())
        gene_f = _parse(F.softmax(self.alphas_f, dim=-1).data.cpu().numpy())

        concat = range(1 + self._steps - self._multiplier, self._steps + 1)

        Genotype = namedtuple('Genotype', 'f1 f1_cat f2 f2_cat f f_cat')

        genotype = Genotype(
            f1=gene_f1, f1_cat=concat,
            f2=gene_f2, f2_cat=concat,
            f=gene_f, f_cat=concat,
        )

        return genotype


class Network_DAG_lat(nn.Module):

    def __init__(self, C, criterion1, criterion2, latency, steps=4, multiplier=4):
        super(Network_DAG_lat, self).__init__()
        self._C = C
        self.l1_loss = criterion1
        self.mse_loss = criterion2
        self._steps = steps
        self._multiplier = multiplier

        self.stem1 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(C*8, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(C*4, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, 1, 3, padding=1, bias=False),
            nn.Tanh()
        )

        self.lat = latency

        self.cell_f1 = CellMIF_lat(C, latency=self.lat['cell_f1'], steps=self._steps, multiplier=self._multiplier)
        self.cell_f2 = CellMIF_lat(C, latency=self.lat['cell_f2'], steps=self._steps, multiplier=self._multiplier)
        self.cell_f = CellMIF_lat(C, latency=self.lat['cell_f'], steps=self._steps, multiplier=self._multiplier)

        self._initialize_alphas()

    def new(self):
        model_new = Network_DAG_lat(self._C, self.l1_loss, self.mse_loss).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, x1, x2):
        f1 = self.stem1(x1)
        f2 = self.stem2(x2)

        weights_f1 = F.softmax(self.alphas_f1, dim=-1)
        weights_f2 = F.softmax(self.alphas_f2, dim=-1)

        f1_out, lat_f1 = self.cell_f1(f1, weights_f1)
        f2_out, lat_f2 = self.cell_f2(f2, weights_f2)

        f_in = torch.cat([f1_out, f2_out], dim=1)

        weights_f = F.softmax(self.alphas_f, dim=-1)
        f_in1 = self.fusion1(f_in)

        f_out, lat_f = self.cell_f(f_in1, weights_f)
        fus = self.fusion2(f_out)

        total_lat = lat_f1 + lat_f2 + lat_f

        return fus, total_lat

    def _loss(self, fus, mri, oth, mri_mask, oth_mask, seg):

        # Pixel Loss
        loss_pixel = self.mse_loss(fus * mri_mask, mri * mri_mask) + self.mse_loss(fus * oth_mask, oth * oth_mask)

        # Gradient Loss
        sobelconv = Sobelxy()
        grad_max = torch.max(sobelconv(mri),sobelconv(oth))
        grad_fus = sobelconv(fus)
        loss_grad = self.l1_loss(grad_fus, grad_max)

        # Segment Loss
        seg = torch.where(seg <= 0, torch.ones_like(seg) * 0.25, torch.ones_like(seg) * 0.75)
        loss_seg = self.l1_loss(fus * seg, mri * seg)
        
        # Total Loss
        loss_total =  loss_pixel + 0.5* loss_grad + 0.05* loss_seg

        return loss_total

    def _initialize_alphas(self): 
        k = sum(1 for i in range(self._steps) for n in range(1 + i)) #node之间总共几根线
        num_ops = len(PRIMITIVES)

        self.alphas_f1 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_f2 = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_f = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_f1,
            self.alphas_f2,
            self.alphas_f
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        gene_f1 = _parse(F.softmax(self.alphas_f1, dim=-1).data.cpu().numpy())
        gene_f2 = _parse(F.softmax(self.alphas_f2, dim=-1).data.cpu().numpy())
        gene_f = _parse(F.softmax(self.alphas_f, dim=-1).data.cpu().numpy())

        concat = range(1 + self._steps - self._multiplier, self._steps + 1)

        Genotype = namedtuple('Genotype', 'f1 f1_cat f2 f2_cat f f_cat')

        genotype = Genotype(
            f1=gene_f1, f1_cat=concat,
            f2=gene_f2, f2_cat=concat,
            f=gene_f, f_cat=concat,
        )

        return genotype

