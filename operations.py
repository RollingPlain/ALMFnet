import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
    'den_conv_1x1': lambda C: DenseBlock(C, C, 1),
    'den_conv_3x3': lambda C: DenseBlock(C, C, 3),
    'den_conv_5x5': lambda C: DenseBlock(C, C, 5),
    'res_conv_1x1': lambda C: ResidualModule(C, C, 1),
    'res_conv_3x3': lambda C: ResidualModule(C, C, 3),
    'res_conv_5x5': lambda C: ResidualModule(C, C, 5),
    'skip_connect': lambda C: FactorizedReduce(C, C),
    'sep_conv_1x1': lambda C: SepConv(C, C, 1, stride = 1, padding = 0),
    'sep_conv_3x3': lambda C: SepConv(C, C, 3, stride = 1, padding = 1),
    'sep_conv_5x5': lambda C: SepConv(C, C, 5, stride = 1, padding = 2),
    'sep_conv_7x7': lambda C: SepConv(C, C, 7, stride = 1, padding = 3),
    'dil_conv_3x3': lambda C: DilConv(C, C, 3, stride = 1, padding = 2, dilation = 2),
    'dil_conv_5x5': lambda C: DilConv(C, C, 5, stride = 1, padding = 4, dilation = 2),
    'ECAattention': lambda C: ECABasicBlock(C),
    'SPAattention':  lambda C: Spatial_BasicBlock(C),
    'NonLocalattention' : lambda C: NLBasicBlock(C),
    'conv_3x1_1x3': lambda C: nn.Sequential(
        nn.Conv2d(C, C, (1, 3), stride=(1, 1), padding=(0, 1), bias=False),
        nn.Conv2d(C, C, (3, 1), stride=(1, 1), padding=(1, 0), bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=False)
    ),
    'conv_5x1_1x5': lambda C: nn.Sequential(
        nn.Conv2d(C, C, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
        nn.Conv2d(C, C, (5, 1), stride=(1, 1), padding=(2, 0), bias=False),
        nn.BatchNorm2d(C),
        nn.ReLU(inplace=False)
    ),
}

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, dilation=1, groups=1, relu=True, bn=False,
                 bias=False):
        super(BasicConv, self).__init__()
        # judge
        stride = 1
        padding = 0
        if kernel_size == 3 and dilation == 1:
            padding = 1
        if kernel_size == 3 and dilation == 2:
            padding = 2
        if kernel_size == 5 and dilation == 1:
            padding = 2
        if kernel_size == 5 and dilation == 2:
            padding = 4
        if kernel_size == 7 and dilation == 1:
            padding = 3
        if kernel_size == 7 and dilation == 2:
            padding = 6
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, dialtions=1):
        super(DenseBlock, self).__init__()
        self.conv1 = BasicConv(C_in, C_in, kernel_size, dilation=dialtions, relu=False)
        self.conv2 = BasicConv(C_in * 2, C_in, kernel_size, dilation=dialtions, relu=False)
        self.conv3 = BasicConv(C_in * 3, C_out, kernel_size, dilation=dialtions, relu=False)

        self.lrelu = nn.PReLU()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        return x3 * 0.333333 + x


class ResidualModule(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, dialtions=1):
        super(ResidualModule, self).__init__()
        self.op = nn.Sequential(
            BasicConv(C_in, C_in, kernel_size, dilation=dialtions, relu=False, groups=C_in),
            nn.Conv2d(C_in, C_in, kernel_size=3, stride=1, padding=2, dilation=2, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.op(x)
        return x + res


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

################################### Channel Attention ############################################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class eca_layer(nn.Module):

    def __init__(self, k_size=3):
      super(eca_layer, self).__init__()
      self.avg_pool = nn.AdaptiveAvgPool2d(1)
      self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
      self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      y = self.avg_pool(x)
      y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      y = self.sigmoid(y)
      y = x * y.expand_as(x)
      return y


class ECABasicBlock(nn.Module):
    def __init__(self, C, kernel =3):
        super(ECABasicBlock, self).__init__()

        self.conv1 = conv3x3(C,C)
        self.conv2 = BasicConv(C,C,kernel,relu=False)
        self.se = eca_layer(k_size=3)
        self.relu = nn.PReLU()

    def forward(self, x):
        out= x = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out

################################### Spatial Attention ############################################
class ChannelPool(nn.Module):
  def forward(self, x):
    return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  
        return x * scale

class Spatial_BasicBlock(nn.Module):

    def __init__(self, C, kernel =3, stride=1):
        super(Spatial_BasicBlock, self).__init__()

        self.conv1 = conv3x3(C, C, stride)
        self.conv2 = BasicConv(C,C,kernel,relu=False)
        self.se = spatial_attn_layer(kernel)
        self.relu = nn.PReLU()

    def forward(self, x):
        out= x = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out

################################### Non-Local Attention ############################################

class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels, bias=False):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                            padding=0, bias=bias)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                            padding=0, bias=bias)
        nn.init.constant_(self.W.weight, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                                padding=0, bias=bias)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                            padding=0, bias=bias)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class NLBasicBlock(nn.Module):
    def __init__(self, C, kernel=3, stride=1,):
        super(NLBasicBlock, self).__init__()

        self.conv1 = conv3x3(C, C, stride)
        self.conv2 = BasicConv(C, C, kernel, relu=False)
        self.se = NonLocalBlock2D(C,C)
        self.relu = nn.PReLU()

    def forward(self, x):
        out= x = self.conv1(x)
        out = self.relu(out)
        out = self.se(out)
        out += x
        out = self.conv2(out)
        out = self.relu(out)
        return out