'''
modified:
--result_path
--pretrain_path
--train_pickle
--test_pickle
root_path of PedesAttrVal (Make_datasets.pedes)
'''
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
print("#####" + str(curPath))
root_path = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
sys.path.append(root_path)

import argparse
import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch import optim
from utils.Logger import Logger
from torch.utils.data import DataLoader

from model.conformer import Conformer  #####

from model.engine import train_conformer, evaluate_conformer

from Datasets.pedes import PedesAttr, PedesAttrVal
from utils.predict import prediction
from Datasets.augmentation import get_transform
from utils.BCELoss import BCELoss
import wandb
wandb.login()


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="", type=str,help='Use the basic model or are there any additional modules?')
    parser.add_argument('--result_path', default="Result", type=str,help='Result directory path')
    parser.add_argument('--pretrain_path', default="Result/Pretrained/Conformer_base_patch16.pth", type=str, help='Pretrained model (.pth)')
    parser.add_argument('--train_pickle', default="Datasets/train_all.pkl", type=str, help='File train')
    parser.add_argument('--val_pickle', default="Datasets/val_all.pkl", type=str, help='File val')
    parser.add_argument('--test_pickle', default="Datasets/val_all.pkl", type=str, help='File test')
    parser.add_argument('--resume_path', default="", type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--sample_size', default=[512, 256], type=int, help='Height and width of inputs')
    parser.add_argument('--scale_size', default=[[512, 256],
                                                 [384, 256], [384, 192],
                                                 [320, 256], [320, 192], [320, 160],
                                                 ], type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--power', default=0.9, type=float, help='Power')
    parser.add_argument('--max_iter', default=2500 * 4000000000000, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight Decay')
    parser.add_argument('--mean', default=[0.485, 0.456, 0.406], type=float, help='Weight Decay')
    parser.add_argument('--std', default=[0.229, 0.224, 0.225], type=float, help='Weight Decay')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=5, type=int, help='Number of total epochs to run')
    parser.add_argument('--print_freq', default=200, type=int, help='print')
    parser.add_argument('--begin_epoch', default=0, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')

    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=True)
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=True)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--predict', action='store_true', help='If true, then predict.')
    parser.set_defaults(predict=True)
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint', default=1, type=int, help='Trained model is saved at every this epochs.')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--norm_value', default=255, type=int, help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--model', default='resnet', type=str, help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth', default=1, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--lr_steps', default=20, type=int)
    parser.add_argument('--dataset', default='PAR', type=str, help='( RAP | PETA | PA100k )')

    parser.add_argument('--supression', default='Join', type=str, help='(Random | Peak | Join)')

    parser.add_argument('--VAL_SPLIT', default='val', type=str)
    parser.add_argument('--TRAIN_SPLIT', default='train', type=str)
    parser.add_argument('--TEST_SPLIT', default='test', type=str)
    parser.add_argument('--TARGETTRANSFORM', default=[])
    parser.add_argument('--SAMPLE_WEIGHT', default='weight', type=str)
    parser.add_argument('--SCALE', default=1, type=int)
    parser.add_argument('--LABEL', default='eval', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.cuda.manual_seed(123)

    opts = parse_opts()
    opts.arch = '{}-{}'.format(opts.model, opts.model_depth)
    # print(opts)
    cudnn.benchmark = True

    ###################       DATASET        ###################
    print('--------------------------------------')
    train_tsfm, valid_tsfm = get_transform(opts)
    train_set = PedesAttr(cfg=opts, path = opts.train_pickle,transform=train_tsfm,
                          target_transform=opts.TARGETTRANSFORM)
    print(len(train_set))
    print('--------------------------------------')
    valid_set = PedesAttr(cfg=opts, path = opts.val_pickle, transform=valid_tsfm)
    print(len(valid_set))
    # test_set = PedesAttrVal(cfg=opts, transform=valid_tsfm)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              sampler=None,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True, )

    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=opts.batch_size,
                              shuffle=False,
                              num_workers=8,
                              pin_memory=True, )

    # test_loader = DataLoader(dataset=test_set,
    #                          batch_size=opts.batch_size,
    #                          sampler=None,
    #                          shuffle=False,
    #                          num_workers=8,
    #                          pin_memory=True)

    if not False:
        train_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                              ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                    ['epoch', 'batch', 'iter', 'loss', 'lr'])

    if opts.nesterov:
        dampening = 0
    else:
        dampening = opts.momentum

    ###################       Loss        ###################
    #
    labels = train_set.label
    label_ratio = labels.mean(0) if opts.SAMPLE_WEIGHT else None
    criterion = BCELoss(sample_weight=label_ratio, scale=opts.SCALE, size_sum=True)
    criterion = criterion.cuda()
    model = Conformer(num_classes=train_set.attr_num)

    # Default load pretrained
    checkpoint = torch.load(opts.pretrain_path, map_location='cpu')
    # Tiếp tục thực hiện các hành động khác ở đây nếu cần thiết
    # print(checkpoint.keys())
    model_dict = model.state_dict()
    

    # pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and not k.startswith(('trans_cls_head', 'conv_cls_head'))}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()
    
    # loaded_params = sum(p.numel() for p in model.parameters() if p.requires_grad and p.is_cuda)
    # print(loaded_params)

    ##################################################################
    print("--------------------")
    print("Building model")
    ##################################################################

    cudnn.benchmark = True
    params = model.parameters()

    labels = train_set.label
    label_ratio = labels.mean(0)  # if cfg.LOSS.SAMPLE_WEIGHT else None  #  SAMPLE_WEIGHT: 'weight'
    criterion = BCELoss(sample_weight=label_ratio, scale=1, size_sum=True)
    criterion = criterion.cuda()

    optimizer = optim.SGD(params, lr=opts.learning_rate,
                          momentum=opts.momentum, dampening=dampening,
                          weight_decay=opts.weight_decay, nesterov=opts.nesterov)

    # load resume
    if opts.resume_path:
        print('loading checkpoint {}'.format(opts.resume_path))
        checkpoint = torch.load(opts.resume_path)
        assert opts.arch == checkpoint['arch']

        opts.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opts.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
    ##################################################################
    print('RUN')
    ##################################################################

    print('-------------------')
    print('1. Prediction')
    print('2. Train')
    print('-------------------')
    choice = input('Enter your choice: ')
    
    if choice == '1' or choice == 1:
        print('Predicting...')
        index = opts.begin_epoch
        prediction(index, test_loader, model, test_set.attribute, opts)
    else:
        print('Training...')
        wandb.init(
            project='PAR-challenge',
            name=opts.model_name,
            config={
                'learning_rate': opts.learning_rate,
                'epochs': opts.n_epochs,
                'Batch Size': opts.batch_size,
                'Momentum': opts.momentum,
                'mean' : opts.mean,
                'Std': opts.std
            })
        for i in range(opts.begin_epoch, opts.n_epochs + 1):
            if not opts.no_train:
                train_conformer(i, train_loader, model, criterion, optimizer,
                                           opts, train_logger, train_batch_logger)
                
            if opts.test:
                evaluate_conformer(i, valid_loader, model, opts)        
        wandb.finish()



