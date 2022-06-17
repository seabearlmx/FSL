import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
import os.path as osp
import logging
from PIL import Image
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from evaluation_multi import compute_mIoU
from evaluation_multi import colorize_mask
#import tensorboardX
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
from options.test_options import TestOptions
import scipy.io as sio
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)


loss_log_file = r'./loss_log.txt'
logging.basicConfig(
    level=logging.INFO,
    format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M',
    filename=loss_log_file,
    filemode='a');


def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)


def main():
    opt = TrainOptions()
    args = opt.initialize()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}

    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    # losses to log
    loss = ['loss_seg_src', 'loss_seg_trg']
    loss_train = 0.0
    loss_val = 0.0
    loss_train_list = []
    loss_val_list = []
    bestIoU = -1.0

    class_weights = Variable(CS_weights).cuda()

    _t['iter time'].tic()

    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        src_img, src_lbl, _, _ = sourceloader_iter.next()                            # new batch source
        trg_img, trg_lbl, _, _ = targetloader_iter.next()                            # new batch target

        # evaluate and update params #####
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda() # to gpu
        src_feats, src_seg_score = model(src_img, lbl=src_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_src = model.loss_seg                                                # get loss
        loss_ent_src = model.loss_ent

        # get target loss, only entropy for backpro
        trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda() # to gpu
        trg_feats, trg_seg_score = model(trg_img, lbl=trg_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_trg = model.loss_seg                                                # get loss
        loss_ent_trg = model.loss_ent

        triger_ent = 0.0
        if i > args.switch2entropy:
            triger_ent = 1.0

        loss_all = loss_seg_src + triger_ent * args.entW * loss_ent_trg     # loss of seg on src, and ent on s and t

        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.detach().cpu().numpy()
        loss_val += loss_seg_trg.detach().cpu().numpy()

        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save( model.state_dict(), os.path.join(args.snapshot_dir, '%s_' % (args.source) + str(i+1) + '.pth') )

            test_opt = TestOptions()
            test_args = test_opt.initialize()
            testloader = CreateTrgDataLoader(test_args)
            model.eval()
            with torch.no_grad():
                for index, batch in enumerate(testloader):
                    if index % 100 == 0:
                        print('%d processd' % index)
                    image, _, name = batch  # 1. get image
                    image = Variable(image).cuda()
                    # forward
                    _, output1 = model(image)
                    output1 = nn.functional.softmax(output1, dim=1)
                    output = output1

                    output = \
                    nn.functional.interpolate(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[
                        0].numpy()
                    output = output.transpose(1, 2, 0)

                    output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                    output_col = colorize_mask(output_nomask)
                    output_nomask = Image.fromarray(output_nomask)
                    name = name[0].split('/')[-1]
                    output_nomask.save('%s/%s' % (test_args.save, name))
                    output_col.save('%s/%s_color.png' % (test_args.save, name.split('.')[0]))
            # compute_mIoU(args.gt_dir, args.save, args.devkit_dir, args.restore_from)
            mIoUs = compute_mIoU(osp.join(test_args.data_dir_target, 'gtFine/val'), test_args.save, 'dataset/cityscapes_list', os.path.join(args.snapshot_dir, '%s_' % (args.source) + str(i+1)))
            mIoU = round(np.nanmean(mIoUs) * 100, 2)
            if mIoU > bestIoU:
                bestIoU = mIoU
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'BestGTA5.pth'))
            print('===> Best mIoU: ' + str(round(np.nanmean(bestIoU), 2)))
            str_bestiou = 'Best mIoU: ' + str(round(np.nanmean(bestIoU), 2))
            logging.info(str_bestiou)
            model.train()
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            msg = {'it': i + 1,
                   'src seg loss': loss_seg_src.data,
                   'trg seg loss': loss_seg_trg.data,
                   'lr': optimizer.param_groups[0]['lr'] * 10000,
                   'iter time': _t['iter time'].diff
                   }
            print('[it %d][src seg loss %.4f][trg seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, loss_seg_trg.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )
            logging.info(msg)

            sio.savemat(args.tempdata, {'src_img':src_img.cpu().numpy(), 'trg_img':trg_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_val   /= args.print_freq
            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            sio.savemat( args.matname, {'loss_train':loss_train_list, 'loss_val':loss_val_list} )
            loss_train = 0.0
            loss_val = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

if __name__ == '__main__':
    main()

