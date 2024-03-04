import torch
import torch.nn as nn
from operations import *
from operations import ReLUConvBN

class CellMIF(nn.Module):

    def __init__(self, C, genotype, type):
        super(CellMIF, self).__init__()

        self.preprocess0 = ReLUConvBN(C, C, 1, 1, 0)

        if type == 'f1':
            op_names, indices = zip(*genotype.f1)
            concat = genotype.f1_cat
        if type == 'f2':
            op_names, indices = zip(*genotype.f2)
            concat = genotype.f2_cat
        if type == 'f':
            op_names, indices = zip(*genotype.f)
            concat = genotype.f_cat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2 + 1 
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0):
        s0 = self.preprocess0(s0)

        states = [s0]
        for i in range(self._steps):
            if i == 0:
                h = states[self._indices[i]] 
                op = self._ops[i]
                s = op(h)
                states += [s]
            else:
                h1 = states[self._indices[2 * i - 1]]
                h2 = states[self._indices[2 * i]]
                op1 = self._ops[2 * i - 1]
                op2 = self._ops[2 * i]
                h1 = op1(h1)
                h2 = op2(h2)
                s = h1 + h2
                states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class Network_DAG(nn.Module):

    def __init__(self, C, genotype):
        super(Network_DAG, self).__init__()
        self._C = C

        self.stem1 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        self.fusion1 = nn.Sequential(
            nn.Conv2d(C * 8, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(C * 4, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=False),
            nn.Conv2d(C, 1, 3, padding=1, bias=False),
            nn.Tanh()
        )

        self.cell_f1 = CellMIF(C, genotype, type='f1')
        self.cell_f2 = CellMIF(C, genotype, type='f2')
        self.cell_f = CellMIF(C, genotype, type='f')

    def forward(self, mri, oth):
        f1 = self.stem1(mri)
        f2 = self.stem2(oth)

        f1_out = self.cell_f1(f1)
        f2_out = self.cell_f2(f2)

        f_in1 = self.fusion1(torch.cat([f1_out, f2_out], dim=1))

        f_out1 = self.cell_f(f_in1)

        fus = self.fusion2(f_out1)

        return fus