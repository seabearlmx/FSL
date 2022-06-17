from model.deeplab import Deeplab
import torch.optim as optim
from model.deeplab_vgg import DeeplabVGG


def CreateModel(args):
    if args.model == 'DeepLab':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=phase)

        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model
        
    if args.model == 'VGG':
        phase = 'test'
        if args.set == 'train' or args.set == 'trainval':
            phase = 'train'
        model = DeeplabVGG(num_classes=args.num_classes, vgg16_caffe_path=args.init_weights, restore_from=args.restore_from,
                        phase=phase, pretrained=True)

        if args.set == 'train' or args.set == 'trainval':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model


def CreateSSLModel(args):
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=args.set)
    elif args.model == 'VGG':
        model = DeeplabVGG(num_classes=args.num_classes, vgg16_caffe_path=args.init_weights,
                           restore_from=args.restore_from,
                           phase=phase, pretrained=True)
    else:
        raise ValueError('The model mush be either deeplab-101 or vgg16-fcn')
    return model

