"""
Implementation of MixMaatch, from paper MixMatch: A Holistic Approach to Semi-Supervised Learning
https://arxiv.org/abs/1905.02249
"""

import os
import time
import numpy as np
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.distributions import Beta
import torch.utils.data as data
from metrics import accuracy
from progress.bar import Bar as Bar

import Mixmatch_CIFAR10 as dataset
from utils import AverageMeter, Logger, WeightEMA, mkdir_p, save_checkpoint
from wrn_28_2 import WideResNet

#options
parser = argparse.ArgumentParser(description='PyTorch MixMatch')

parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--base_model', default='wrn-28-2', type=str, help='name of base model')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_dir', default='data', type=str, help='data directory')
parser.add_argument('--num_classes', default=10, type=int, help='number of classes')
parser.add_argument('--num_labeled', default=250, type=int, help='number of labeled data')

parser.add_argument('--batch_size', default=32, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float, help='initial learning rate')
parser.add_argument('--ema-decay', default=0.999, type=float, help='ema variable decay rate')
parser.add_argument('--epochs', default=1024, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--num_val', default=5000, type=int, help='number of validation data')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')

parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', type=str, help='folder to output images and model checkpoints')
parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')

parser.add_argument('--train-iteration', default=1024, type=int, help='number of iteration per epoch')
parser.add_argument('--temp', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=0.75, type=float, help='alpha for beta distribution')
parser.add_argument('--lambda-u', default=100, type=float, help='weight for unlabeled loss')
parser.add_argument('--T', default=0.5, type=float, help='pseudo label temperature')
parser.add_argument('--K', default=2, type=int, help='number of augmentations per unlabeled example')

parser.add_argument('--toperr', default='TOP1', type=str, help='TOP1 or TOP5')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

#gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

#random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

#one hot label function
def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1), 1)

#shrpening function
def sharpen(x, T):
    x = x**(1/T)
    return x / x.sum(dim=1, keepdim=True)

#label guessing function
def label_guessing(model, u, K):
    for i in range(K):
        u[i] = model(u[i])
    return u

#mixup function
def mixup(x1, x2, x1_label, x2_label, alpha):
    lam = Beta(alpha, alpha) 
    lam = torch.amax([lam, 1-lam])
    x = lam * x1 + (1 - lam) * x2
    y = lam * x1_label + (1 - lam) * x2_label
    return x, y

#agumentation function


def MixMatch(x, y, u, T, K, alpha, num_classes, model):
    #one hot label of x
    y_oh = y.cuda()
    # label data agumentation
    xhat = x.cuda()
    #unlabel data agumentation * K
    uhat = np.aaray(u)
    #label guessing
    qb =torch.mean(label_guessing(model, uhat, K), axis=0)
    #sharpening
    q = sharpen(qb, T)
    #mixup
    W_x = torch.randperm(xhat, uhat, dim=0)
    W_y = torch.randperm(y_oh, q,dim=0)
    X, X_label = mixup(xhat, W_x[:len(xhat)], y_oh, W_y[:len(xhat)], alpha)
    U, U_label = mixup(uhat, W_x[len(xhat):], q, W_y[len(xhat):], alpha)
    return X, X_label, U, U_label


class MixMatchLoss(nn.Module):
    def __init__(self, lam_u=100):
        self.lam_u = lam_u
        self.x_loss = torch.nn.CrossEntropyLoss()
        self.u_loss = torch.nn.MSELoss()
        super(MixMatchLoss, self).__init__()

    def forward(self, x, u, p, q):
        x_criteroion = self.x_loss(x, p)
        u_criterion = self.u_loss(u, q)
        return x_criteroion + self.lam_u * u_criterion


def train(labeled_trainloader, 
          unlabeled_trainloader,
          model,
          optimizer,
          ema_optimizer,
          num_classes=10,
          use_cuda = True):
    
    #switch to train mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)


    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)
        try:
            inputs_u= next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u= next(unlabeled_train_iter)
        
        data_time.update(time.time() - end)


        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        
        #mixmatch
        X, X_label, U, U_label = MixMatch(x=inputs_x, y=targets_x, u=inputs_u,
                                           T=args.T, K=args.K, alpha=args.alpha, 
                                           num_classes=num_classes,
                                           model=model)
        
        Loss = MixMatchLoss(X, X_label, U, U_label)

        #record loss
        losses.update(Loss.item(), inputs_x.size(0))

        #backward
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        ema_optimizer.step()

        #update EMA
        ema_optimizer.step()

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg)


def validate(val_loader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            #forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
        
        if mode == top1:
            return (losses.avg, top1.avg)
        elif mode == top5:
            return (losses.avg, top5.avg)
        else:
            raise ValueError('mode must be top1 or top5')

#main
best_acc = 0

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing cifar10')
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10_set('./data', args.num_labeled, transform_train, transform_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #base model
    print('base model : WideResNet-28-2')

    def get_model(ema=False):
        model = WideResNet(num_classes=10)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()

        return model
    
    model = get_model()
    ema_model = get_model(ema=True)

    train_criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay, lr=args.lr)
    start_epoch = 0

    cudnn.benchmark = True
    print('   Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []

    # Resume
    title = 'noisy-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])


    #Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, use_cuda)
        train_loss, train_top1 = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_top1 = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_top1 = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.train_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_top1, step)
        writer.add_scalar('accuracy/val_acc', val_top1, step)
        writer.add_scalar('accuracy/test_acc', test_top1, step)

        # append logger file
        logger.append([train_loss, train_loss, train_top1, val_loss, val_top1, test_loss, test_top1])

        # save model
        is_best = val_top1 > best_acc
        best_acc = max(val_top1, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'top1': val_top1,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_top1)



    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(torch.mean(test_accs[-20:]))


if __name__ == '__main__':
    main()