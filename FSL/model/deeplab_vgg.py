import numpy as np
import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F


class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, phase='train', vgg16_caffe_path=None, pretrained=False, restore_from=None,):
        self.phase = phase
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))
        elif restore_from is not None:
            vgg.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23, 25, 27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features = nn.Sequential(*([features[i] for i in range(len(features))] + [ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)]))

        self.classifier = Classifier_Module(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

    def forward(self, x, lbl=None, weight=None, ita=1.5):
        _, _, h, w = x.size()
        # print(x.size())
        x = self.features(x)
        feats = x
        x = self.classifier(x)  # produce segmap 2
        if self.phase == 'train':
            P = F.softmax(x, dim=1)        # [B, 19, H, W]
            logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
            PlogP = P * logP               # [B, 19, H, W]
            ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
            ent = ent / 2.9444         # chanage when classes is not 19
            # compute robust entropy
            ent = ent ** 2.0 + 1e-8
            ent = ent ** ita
            self.loss_ent = ent.mean()

            x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            if lbl is not None:
                self.loss_seg = self.CrossEntropy2d(x, lbl, weight=weight)
        return feats, x

    def optim_parameters(self, args):
        return self.parameters()

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10

    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        # print(predict.size())
        # print(target.size())
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)

        return loss
