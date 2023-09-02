
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
from torch import optim
import utils.visualization as visual
from utils import data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.utils import clip_gradient, adjust_lr
from utils.metrics import Evaluator

from network.CGNet import HCGMNet,CGNet

import time
start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)

    length = 0
    st = time.time()
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A,B)
        loss = criterion(preds[0], Y)  + criterion(preds[1], Y)
        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)
        
        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU, Pre, Recall, F1))
    print("Strat validing!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)

            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number') #修改这里！！！
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size') #修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')  #修改这里！！！
    parser.add_argument('--data_name', type=str, default='WHU', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='CGNet',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    if opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    if opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = '/data/chengxi.han/data/LEVIR CD Dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/LEVIR CD Dataset256/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/data/chengxi.han/data/Building change detection dataset256/train/'
        opt.val_root = '/data/chengxi.han/data/Building change detection dataset256/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
        opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    elif opt.data_name == 'DSIFN':
        opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
        opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    elif opt.data_name == 'SYSU':
        opt.train_root = '/data/chengxi.han/data/SYSU-CD/train/'
        opt.val_root = '/data/chengxi.han/data/SYSU-CD/val/'
    elif opt.data_name == 'S2Looking':
        opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
        opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'

    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)

    if opt.model_name == 'HCGMNet':
        model = HCGMNet().cuda()
    elif opt.model_name == 'CGNet':
        model = CGNet().cuda()


    criterion = nn.BCEWithLogitsLoss().cuda()


    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")
    # args = parser.parse_args()
    # print('现在的数据是：',args.data_name)


    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        Eva_train.reset()
        Eva_val.reset()
        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer, opt.epoch)
        lr_scheduler.step()
        # print('现在的数据是：', args.data_name)

end=time.time()
print('程序训练train的时间为:',end-start)