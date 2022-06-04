import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from functools import reduce
import numpy as np

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet32(nn.Module):
    def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5]):
        super(ResNet32, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class WiderBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WiderBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WiderNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(WiderNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = WiderBasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = WiderNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = WiderNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = WiderNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, in_size=1,hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(in_size, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)



class NRT(nn.Module):
    def __init__(self, num_cut,temperature=0.1):
        super(NRT, self).__init__()
        self.num_leaf = np.prod(np.array(num_cut) + 1)
        self.leaf_score =  torch.nn.Parameter(torch.rand([self.num_leaf, 1]))
        self.cut_points_list = [torch.nn.Parameter(torch.rand([i])).cuda() for i in num_cut]
        self.temperature = temperature

    def kron_prod(self, a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res

    def to_sprase(self, a):
        a = torch.where((a==torch.max(a))==True, 1, 0)*a
        idx = torch.nonzero(a).T
        data = a[idx[0],idx[1]]
        coo_a = torch.sparse_coo_tensor(idx, data, a.shape)
        return coo_a
    
    def kron_prod_sparse(self, a, b):
        batch = []
        for i in range(a.shape[0]):
            sprase_ai,sprase_bi = self.to_sprase(a[i].unsqueeze(1)),self.to_sprase(b[i].unsqueeze(0))
            y = torch.sparse.mm(sprase_ai,sprase_bi)
            batch.append(y.to_dense().view(-1).unsqueeze(0))
        return torch.cat(batch)
    
    def soft_bin(self, x,cut_points):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1]).cuda()
        cut_points, _ = torch.sort(cut_points) 
        b = torch.cumsum(torch.cat([torch.zeros([1]).cuda(), - cut_points], 0),0)
        h = torch.matmul(x, W) + b
        ht = F.softmax(h / self.temperature,dim=1)
        return ht

    def forward(self, x):
        x = reduce(self.kron_prod,map(lambda z: self.soft_bin(x[:, z[0]:z[0] + 1], z[1]), enumerate(self.cut_points_list)))
        x = torch.matmul(x, self.leaf_score)
        return torch.sigmoid(x)



class PRUNNRT(nn.Module):
    def __init__(self,num_feature=9,temperature = 0.1, sub_rate = 0.2,add_rate = 0.1, record_num = 5, mask_max=4):
        super(PRUNNRT, self).__init__()
        self.mask_max = mask_max
        self.num_feature = num_feature
        self.num_cut = [self.mask_max]*self.num_feature
        self.num_leaf = np.prod(np.array(self.num_cut) + 1)
        self.leaf_score =  torch.nn.Parameter(torch.rand([self.num_leaf, 1]))
        self.cut_points_list = [torch.nn.Parameter(torch.rand([cut])) for i,cut in enumerate(self.num_cut)]
        self.bn = nn.BatchNorm1d(self.num_feature)
        self.temperature = temperature
        
        self.mask = [[0]]*self.num_feature
        self.sub_rate = sub_rate
        self.add_rate = add_rate
        
        self.record_num = record_num
        self.cut_points_five = {}
        for i in range(self.num_feature):
            self.cut_points_five[i] = torch.zeros([len(self.mask[i]),self.record_num])
        
    def record_points(self,epoch):
        for i in range(self.num_feature):
            for j in range(self.cut_points_five[i].shape[0]):
                self.cut_points_five[i][j][epoch%self.record_num] = self.cut_points_list[i][j]

    def statistics(self):
        for i in range(self.num_feature):
            flag_number = torch.nonzero(self.cut_points_five[i][0]).size(0)==self.record_num 
            max_five_i = torch.max(self.cut_points_five[i],dim=1).values
            min_five_i = torch.min(self.cut_points_five[i],dim=1).values
            mean_five_i = torch.min(max_five_i-min_five_i)

            if flag_number and mean_five_i<self.add_rate:
                self.up_update(i)
            if len(self.mask[i])>1 :
                # near
                cut_point = self.cut_points_list[i][self.mask[i]].unsqueeze(1)
                cut_point_t = cut_point.T
                cut_point_abs = torch.abs(cut_point - cut_point_t)
                cut_point_abs = torch.where(cut_point_abs == 0.0, torch.ones_like(cut_point_abs), cut_point_abs)
                min_local, min_value = torch.argmin(cut_point_abs).item(),torch.min(cut_point_abs)
                x, y = int(min_local/len(self.mask[i])),int(min_local%len(self.mask[i]))
                
                if min_value<self.sub_rate:
                    self.dw_update(i,x,y)
                
    def up_update(self,i):
        if len(self.mask[i])<self.mask_max:
            pool = [x for x in range(self.mask_max) if x not in self.mask[i]]
            self.mask[i].append(pool[0])
            self.cut_points_five[i] = torch.zeros([len(self.mask[i]),self.record_num])
        
    def dw_update(self,i,x,y):
        self.mask[i] = [x for x in self.mask[i] if x!=self.mask[i][y]]
        self.cut_points_five[i] = torch.zeros([len(self.mask[i]),self.record_num])
        
        
    def kron_prod(self, a, b):
        res = torch.einsum('ij,ik->ijk', [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res
    
    def cut_bin(self, x,cut_points):
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1]).cuda()
        cut_points, _ = torch.sort(cut_points) 
        b = torch.cumsum(torch.cat([torch.zeros([1]), - cut_points], 0),0).cuda()
        h = torch.matmul(x, W) + b
        h = F.softmax(h / self.temperature,dim=1)
        return h

    def forward(self, x):
        x = self.bn(x)
        soft_bin_list = []
        for i,cut_point in enumerate(self.cut_points_list):
            cut_point = cut_point[self.mask[i]]
            bin = self.cut_bin(x[:, i:i+1],cut_point)
            add_bin = torch.zeros([x.shape[0],self.mask_max-bin.shape[1]+1]).cuda()
            soft_bin_list.append(torch.cat((bin,add_bin),dim=1))
            
        x = reduce(self.kron_prod,soft_bin_list)
        x = torch.matmul(x, self.leaf_score)
        return torch.sigmoid(x)