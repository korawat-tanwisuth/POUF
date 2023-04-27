import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils
from tllib.modules.classifier import Classifier
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.metric import accuracy
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
from tllib.utils.data import ForeverDataIterator
import clip
from clip_classifier import ClipClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    clip_model, preprocess = clip.load(args.arch, device)
    clip_model = clip_model.float()
    print(preprocess)

    # Data loading code 
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = preprocess 

    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size_target,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)


    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in args.class_names]).to(device)

    # create model
    print("=> using model '{}'".format(args.arch))
    pool_layer = nn.Identity()
    bottleneck_dim = clip_model.visual.output_dim

    classifier = ClipClassifier(clip_model, args.class_names, num_classes, learn_prompt=args.learn_prompt, bottleneck = None, pool_layer=pool_layer, finetune=not args.scratch, device=device).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(prompt_lr=args.prompt_lr), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print(lr_scheduler.get_lr())
       
        train(train_target_iter, classifier, optimizer,
            lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
            best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()

def get_entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def compute_im_loss(logits):
    softmax_out = nn.Softmax(dim=1)(logits)
    entropy_loss = torch.mean(get_entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
    im_loss = entropy_loss - gentropy_loss
    return im_loss

def compute_transport_loss(logits, sim_t):
    s_dist = F.softmax(logits, dim=1)
    t_dist = F.softmax(logits, dim=0)
    cost = 1 - sim_t
    s_cost = (cost * s_dist).sum(1).mean()
    t_cost = (cost * t_dist).sum(0).mean()
    return s_cost + t_cost
 
def train(train_target_iter: ForeverDataIterator,
          model: ClipClassifier, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    transfer_losses = AverageMeter('Transfer Loss', ':6.2f')
    im_losses = AverageMeter('IM Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, transfer_losses, im_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_t, _ = next(train_target_iter)[:2]

        x_t = x_t.to(device)
        _ = _.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        sim_t, f_t = model(x_t)
        
        
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * sim_t
         
        # compute loss
        transfer_loss =  compute_transport_loss(logits, sim_t) 
        im_loss = compute_im_loss(logits)
        loss = args.trade_off * transfer_loss + args.lambda_im * im_loss
 
        cls_acc = accuracy(sim_t, _)[0]

        transfer_losses.update(transfer_loss.item(), x_t.size(0))
        im_losses.update(im_loss.item(), x_t.size(0))
        cls_accs.update(cls_acc.item(), x_t.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source Only for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.48145466, 0.4578275, 0.40821073), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.26862954, 0.26130258, 0.27577711), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16',
                        choices=['ViT-B/16'],
                        help='backbone architecture')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-bt', '--batch-size-target', default=96, type=int,
                        metavar='N',
                        help='mini-batch size (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0002, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--lambda-im', default=0.3, type=float,
                        help='the trade-off hyper-parameter for im loss')
    parser.add_argument('--learn-prompt', action='store_true',
                        help='whether to learn prompt')
    parser.add_argument('-plr', '--prompt-lr', default=5e-4, type=float,
                        help='prompt learning rate (default: 5e-4)')
  
    args = parser.parse_args()
    main(args)
