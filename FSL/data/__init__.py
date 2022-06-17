import numpy as np
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from data.synthia_dataset import SYNDataSet
from data.augmentations import *

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}
cs_size_test = {'cityscapes': (1344,576)}
data_aug = Compose([ColorJitter(), RandomGaussianBlur()])

def CreateSrcDataLoader(args):
    if args.source == 'gta5':
        source_dataset = GTA5DataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'], 
                                      resize=image_sizes['gta5'] ,mean=IMG_MEAN, augmentations=data_aug,
                                      max_iters=args.num_steps * args.batch_size )
    elif args.source == 'synthia':
        source_dataset = SYNDataSet( args.data_dir, args.data_list, crop_size=image_sizes['cityscapes'],
                                      resize=image_sizes['synthia'] ,mean=IMG_MEAN, augmentations=data_aug,
                                      max_iters=args.num_steps * args.batch_size )
    else:
        raise ValueError('The source dataset mush be either gta5 or synthia')
    
    source_dataloader = data.DataLoader( source_dataset, 
                                         batch_size=args.batch_size,
                                         shuffle=True, 
                                         num_workers=args.num_workers, 
                                         pin_memory=True )    
    return source_dataloader

def CreateTrgDataLoader(args):
    if args.set == 'train':
        target_dataset = cityscapesDataSetLabel( args.data_dir_target, 
                                                 args.data_list_target, 
                                                 crop_size=image_sizes['cityscapes'], 
                                                 mean=IMG_MEAN,
                                                 max_iters=args.num_steps * args.batch_size, 
                                                 set=args.set )
    else:
        target_dataset = cityscapesDataSet( args.data_dir_target,
                                            args.data_list_target,
                                            crop_size=cs_size_test['cityscapes'],
                                            mean=IMG_MEAN,
                                            set=args.set )

    if args.set == 'train':
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True )
    else:
        target_dataloader = data.DataLoader( target_dataset,
                                             batch_size=1, 
                                             shuffle=False, 
                                             pin_memory=True )

    return target_dataloader



def CreateTrgDataSSLLoader(args):
    target_dataset = cityscapesDataSet( args.data_dir_target, 
                                        args.data_list_target,
                                        crop_size=image_sizes['cityscapes'],
                                        mean=IMG_MEAN, 
                                        set=args.set )
    target_dataloader = data.DataLoader( target_dataset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         pin_memory=True )
    return target_dataloader



def CreatePseudoTrgLoader(args):
    target_dataset = cityscapesDataSetSSL( args.data_dir_target,
                                           args.data_list_target,
                                           crop_size=image_sizes['cityscapes'],
                                           mean=IMG_MEAN,
                                           max_iters=args.num_steps * args.batch_size,
                                           set=args.set,
                                           label_folder=args.label_folder )

    target_dataloader = data.DataLoader( target_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True )

    return target_dataloader

