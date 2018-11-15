import torch.nn as nn
import numpy as np
import math

from util.bernoulli_igo import BernoulliIGO
from util.im2col import im2Col, col2Im, filtering
import util.filters as f
import util.weight as W


class MaxOrMinFilterProcess():
    def __init__(self, ksize, max):
        self.ksize = ksize
        self.max = max

    def __call__(self, img):
        col = im2Col(img, self.ksize, stride=1, pad=(self.ksize-1)//2)
        col = col.view(-1, self.ksize * self.ksize)

        if self.max:
            _col = torch.max(col,1)[0].view(img.size(2) * img.size(3), -1)
        else:
            _col = torch.min(col,1)[0].view(img.size(2) * img.size(3), -1)

        result = col2Im(_col, img, 1)
        return result


class FilterProcess():
    def __init__(self, filter):
        self.filter = filter
        if self.filter is not None:
            self.ksize = self.filter.size()
            self.px = (self.ksize[1] - 1) // 2
            self.py = (self.ksize[0] - 1) // 2

    def __call__(self, img):
        if self.filter is None:
            return img

        self.filter = self.filter.cuda(img.device)
        col = im2Col(img, self.ksize, stride=1, pad=(self.px, self.py))
        filtered_col = filtering(img.size(), col, self.filter)
        result = col2Im(filtered_col, img, 1)
        return result


class Binarization():
    def __init__(self, t):
        self.t = t

    def __call__(self, img):
        result = torch.empty_like(img)
        result[img < self.t] = 0.
        result[img >= self.t] = 1.
        return result

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Cifar(nn.Module):
    def __init__(self, in_channel, block, layers, num_classes=10, lam=2, eta_coeff=1., init_theta=0.5, all_filter=False):
        super(ResNet_Cifar, self).__init__()
        self.inchannel = in_channel
        self.lam = lam
        self.all_filter = all_filter
        self.preprocess = ['no_op', 'max_3x3', 'max_5x5', 'min_3x3', 'min_5x5',
                           'ave_3x3', 'ave_5x5', 'sobel_h', 'sobel_w', 'lap_3x3',
                           'lap_5x5', 'gaus_3x3', 'gaus_5x5', 'shap_4', 'shap_8',
                           'bin_0.5']
        self.d = len(self.preprocess) * self.inchannel
        self.inplanes = 16
        self.conv1 = nn.Conv2d(self.d, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)


        func0 = FilterProcess(None) # no operation
        func1 = MaxOrMinFilterProcess(3, True)
        func2 = MaxOrMinFilterProcess(5, True)
        func3 = MaxOrMinFilterProcess(3, False)
        func4 = MaxOrMinFilterProcess(5, False)
        func5 = FilterProcess(f.ave_3x3)
        func6 = FilterProcess(f.ave_5x5)
        func7 = FilterProcess(f.sobel_h)
        func8 = FilterProcess(f.sobel_w)
        func9 = FilterProcess(f.lap_3x3)
        func10 = FilterProcess(f.lap_5x5)
        func11 = FilterProcess(f.gaus_3x3)
        func12 = FilterProcess(f.gaus_5x5)
        func13 = FilterProcess(f.shap_4)
        func14 = FilterProcess(f.shap_8)
        func15 = Binarization(0.5)


        self.preprocess_listR = [func0, func1, func2, func3, func4, func5, func6,
                                    func7, func8, func9, func10, func11, func12, func13,
                                    func14, func15]
        self.preprocess_listG = [func0, func1, func2, func3, func4, func5, func6,
                                    func7, func8, func9, func10, func11, func12, func13,
                                    func14, func15]
        self.preprocess_listB = [func0, func1, func2, func3, func4, func5, func6,
                                    func7, func8, func9, func10, func11, func12, func13,
                                    func14, func15]

        init_theta = init_theta * np.ones(self.d)

        non_inc_f = W.SelectionNonIncFunc(threshold=0.25, negative_weight=True)
        w = W.QuantileBasedWeight(non_inc_f=non_inc_f, tie_case=True, normalization=False, min_problem=True)
        self.igo = BernoulliIGO(d=self.d,weight_func=w,theta=init_theta,eta=eta_coeff / self.d, theta_max=1. - 1. / self.d,
                                    theta_min=1. / self.d)
        self.arch = self.igo.sampling_model().sampling(self.lam)

        for m in self.modules():
            if isinstance(nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img, train):
        temp = img
        result = _result = None
        lam = self.lam if train else 1

        for index in range(lam):
            mask[index] = np.where(self.arch[index]>0)[0]
            if self.all_filter:
                mask[index] = np.ones(self.d)

            for n in range(self.d):
                if n in mask[index]:
                    if n % 3 == 0:
                        temp[:,0:1,:,:] = self.preprocess_listR[n//3](img[:,0:1,:,:])
                    elif n % 3 == 1:
                        temp[:,1:2,:,:] = self.preprocess_listG[n//3](img[:,1:2,:,:])
                    else:
                        temp[:,2:3,:,:] = self.preprocess_listB[n//3](img[:,2:3,:,:])
                else:
                    if n % 3 == 0:
                        temp[:,0:1,:,:] = torch.zeros_like(img[:,0:1,:,:])
                    elif n % 3 == 1:
                        temp[:,1:2,:,:] = torch.zeros_like(img[:,1:2,:,:])
                    else:
                        temp[:,2:3,:,:] = torch.zeros_like(img[:,2:3,:,:])

                _result = torch.cat([_result, temp], dim=1) if _result is not None else temp
            temp = img
            result = _result
            _result = None
        result = torch.cat([result, _result], dim=0)
        x = result

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def update_igo(self, losses, train, batchsize=64):
    if train:
        losses = np.array(losses)
        evals = losses.reshape(self.lam, self.batchsize).mean(axis=1)
        self.igo.update(self.arch, evals)
        self.arch = self.igo.sampling_model().sampling(self.lam)
    else:
        #print(self.igo.theta)
        self.arch = self.igo.sampling_model().mle_sampling(batchsize)
    return

def update_arch(self):
    self.arch = self.igo.sampling_model().sampling(self.lam)
    return

def ResNetSelecter(n_layer, in_channel=3):
    if n_layer in [8, 12, 20, 32, 44, 56, 110]:
        # expected insize = 32
        if n_layer == 8:
            layers = [1, 1, 1]
        elif n_layer == 12: #14?
            layers = [2, 2, 2]
        elif n_layer == 20:
            layers = [3, 3, 3]
        elif n_layer == 32:
            layers = [5, 5, 5]
        elif n_layer == 44:
            layers = [7, 7, 7]
        elif n_layer == 56:
            layers = [9, 9, 9]
        elif n_layer == 110:
            layers = [18, 18, 18]
        model = ResNet_Cifar(in_channel, BasicBlock, layers)
        return model
    '''
    elif n_layer in [18, 34]:
        # expected insize = 224
        if n_layer == 18:
            layers = [2, 2, 2, 2]
        elif n_layer == 34:
            layers = [3, 4, 6, 3]
        model = ResNet(in_channel, BasicBlock, layers)
        return model

    elif n_layer in [50, 101, 152]:
        if n_layer == 50:
            layers = [3, 4, 6, 3]
        elif n_layer == 101:
            layers = [3, 4, 23, 3]
        elif n_layer == 152:
            layers = [3, 8, 36, 3]
        model = ResNet(in_channel, Bottleneck, layers)
        return model
    '''
