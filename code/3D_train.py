import argparse
import logging
import os
import random
import shutil
import sys
import time
from typing import Tuple, Union, List
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.transforms import Compose
from dynamic_network.architectures.unet import PlainConvUNet
from dataloaders.dataset import threeDDataSet
from utils import losses
from utils.tools import get_training_transforms
from utils.loss.compound_losses import DC_and_CE_loss
from utils.loss.dice import MemoryEfficientSoftDiceLoss
from val_3D import test_single_volume
from utils.bezier_curve import nonlinear_transformation
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='path/to/training/dataset', help='Path to dataset')
parser.add_argument('--exp', type=str,
                    default='exp/model/path', help='Experiment name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='Model name')
parser.add_argument('--max_iterations', type=int,
                    default=60000, help='Maximum number of iterations to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size per GPU')
parser.add_argument('--deterministic', type=int, default=1,
                    help='Deterministic training')
parser.add_argument('--deep_supervision', type=bool, default=False,
                    help='Deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='Base learning rate for the segmentation network')
parser.add_argument('--patch_size', type=list, default=[128, 128, 128],
                    help='Patch size for the network input')
parser.add_argument('--early_stop_patient', type=float, default=10000,
                    help='num for early stop patient')
parser.add_argument('--nonlinear_transform', type=bool, default=False,
                    help='Number of labeled data samples')
parser.add_argument('--pretrained_path', type=str, default=None, help='Path to the pretrained model')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--labeled_num', type=int, default=None,
                    help='Number of labeled data samples')
parser.add_argument('--validation_iterations', type=int, default=250,
                    help='Number of labeled data samples')
args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    enable_deep_supervision = args.deep_supervision
    early_stop_patient = args.early_stop_patient
    model = PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision = enable_deep_supervision
    )
    model = model.cuda()
    if args.pretrained_path is not None:
        model.load_state_dict(torch.load(args.pretrained_path))
        logging.info(f"Loaded pretrained model from {args.pretrained_path}")
    rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
    mirror_axes = (0, 1, 2)
    train_transforms = get_training_transforms(patch_size=np.array(args.patch_size), deep_supervision_scales=None, 
                                              do_dummy_2d_data_aug=False, rotation_for_DA=rotation_for_DA,
                                                 mirror_axes=mirror_axes, do_spatial_transform=True)
    db_train = threeDDataSet(base_dir=train_data_path,
                             split='train',
                             num=args.labeled_num,
                             transform=train_transforms)
    db_val = threeDDataSet(base_dir=train_data_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn, prefetch_factor=3, persistent_workers=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    loss_dc_ce = DC_and_CE_loss({'batch_dice': True,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
    ce_loss = CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda') 

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    no_improvement_counter = 0.0
    save_best = None
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            if args.nonlinear_transform:
                volume_batch = nonlinear_transformation(volume_batch)
            with torch.amp.autocast('cuda'):
                outputs, _  = model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                pseudo_outputs = torch.argmax(outputs_soft.detach(), dim=1, keepdim=False)
                outputs_soft = torch.softmax(outputs, dim=1)
                pseudo_supervision = ce_loss(outputs, pseudo_outputs)
                # loss = loss_dc_ce(outputs, label_batch)
                loss = pseudo_supervision
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            logging.info(
                'iteration %d : learning rate %f : loss : %f' % (iter_num, lr_, loss.item())
            )
            if iter_num > 0 and iter_num % args.validation_iterations == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model,
                        classes=num_classes, patch_size=args.patch_size, deep_supervision=enable_deep_supervision)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                if performance > best_performance:
                    if save_best is not None and os.path.exists(save_best):
                        os.remove(save_best)
                        logging.info(f"Deleted previous best model: {save_best}")
                    best_performance = performance
                    save_best = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                        iter_num, round(best_performance, 4)))
                    save_mode_path = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format('UNet'))
                    # Save the current best model
                    torch.save(model.state_dict(), save_best)
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"Saved new best model: {save_best}")
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += args.validation_iterations
                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' %
                    (iter_num, performance, np.mean(metric_list, axis=0)[1])
                )
                model.train()
            if iter_num >= max_iterations:
                iterator.close()
                break
        if iter_num >= max_iterations:
            break
        if no_improvement_counter >= early_stop_patient:
            logging.info('No improvement in Validation mean_dice for {} iterations. Early stopping...'.format(early_stop_patient))
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    a = args.labeled_num
    if a==None:
        a = 'source'
    snapshot_path = "../model/{}_{}".format(args.exp, a)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
