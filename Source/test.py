'''
2021/2/3

Guowang Xie

'''
import os
import argparse
import torch
from torch.autograd import Variable

import warnings
import time
import re

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

import utilsV3 as utils

from dataloader import PerturbedDatastsForFiducialPoints_pickle_color_v2_v2

from loss import Losses

def train(args):
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(args.resume).group(0)
        reslut_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w')

    # Setup Dataloader
    data_path = args.data_path_train
    data_path_validate = args.data_path_validate
    data_path_test = args.data_path_test

    print(args)
    print(args, file=reslut_file)

    n_classes = 2

    model = FiducialPoints(n_classes=n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #

    if args.parallel is not None:
        device_ids = list(map(int, args.parallel))
        args.gpu = device_ids[0]
        if args.gpu < 8:
            torch.cuda.set_device(args.gpu)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        warnings.warn('no gpu , go sleep !')
        exit()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.8, weight_decay=1e-12)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=1e-10)
    else:
        assert 'please choice optimizer'
        exit('error')

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    loss_fun_classes = Losses(classify_size_average=True, args_gpu=args.gpu)
    loss_fun = loss_fun_classes.loss_fn4_v5_r_4   # *
    # loss_fun = loss_fun_classes.loss_fn4_v5_r_3   # *

    loss_fun2 = loss_fun_classes.loss_fn_l1_loss

    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, model=model, \
                            reslut_file=reslut_file, n_classes=n_classes, optimizer=optimizer, \
                            loss_fn=loss_fun, loss_fn2=loss_fun2, data_loader=PerturbedDatastsForFiducialPoints_pickle_color_v2_v2, \
                            data_path=data_path, data_path_validate=data_path_validate, data_path_test=data_path_test, data_preproccess=False)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    ''' load data '''
    FlatImg.loadTestData()

    train_time = AverageMeter()
    losses = AverageMeter()

    FlatImg.lambda_loss = 1
    FlatImg.lambda_loss_segment = 0.01
    FlatImg.lambda_loss_a = 0.1
    FlatImg.lambda_loss_b = 0.001
    FlatImg.lambda_loss_c = 0.01

    scheduler = torch.optim.lr_scheduler.MultiStepLR(FlatImg.optimizer, milestones=[40, 90, 150, 200], gamma=0.5)

    epoch_start = checkpoint['epoch'] if args.resume is not None else 0

    if args.schema == 'validate':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t_all')
        exit()
    elif args.schema == 'test':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t_all')
        FlatImg.testModelV2GreyC1_index(epoch, train_time, ['36_2 copy.png', '17_1 copy.png', '22_2 copy.png', '29_2 copy.png'])
        exit()
    elif args.schema == 'eval':
        FlatImg.evalData(is_shuffle=True)

        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.evalModelGreyC1(epoch, is_scaling=False)
        exit()

    m, s = divmod(train_time.sum, 60)
    h, m = divmod(m, 60)
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))
    print("All Train Time : %02d:%02d:%02d\n" % (h, m, s), file=reslut_file)

    reslut_file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='Document-Dewarping-with-Control-Points',
                        help='Architecture')

    parser.add_argument('--img_shrink', nargs='?', type=int, default=None,
                        help='short edge of the input image')

    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimization')

    parser.add_argument('--l_rate', nargs='?', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')            # python segmentation_train.py --resume=./trained_model/fcn8s_pascla_2018-8-04_model.pkl

    parser.add_argument('--print-freq', '-p', default=60, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--data_path_train', default='./dataset/unwarp_new/train/fiducial1024/fiducial1024_v1/color/', type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_validate', default='./dataset/unwarp_new/train/fiducial1024/fiducial1024_v1/validate/', type=str,
                        help='the path of validate images.')  # validate image path

    parser.add_argument('--data_path_test', default='./dataset/shrink_1024_960/crop/', type=str,
                        help='the path of test images.')  # test image path

    parser.add_argument('--output-path', default='./flat/', type=str,
                        help='the path is used to  save output --img or result.')  # GPU id ---choose the GPU id that will be used

    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')       # train  validate

    parser.set_defaults(resume='/gwxie/project/result/flat/2021-02-03/2021-02-03 16:15:55/143/2021-02-03 16:15:55flat_img_by_fiducial_points-fiducial1024_v1.pkl')

    parser.add_argument('--parallel', default='2', type=list,
                        help='choice the gpu id for parallel ')

    args = parser.parse_args()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- not exist')

    if args.data_path_test is None:
        raise Exception('-- No test path')
    else:
        if not os.path.isfile(args.data_path_test):
            raise Exception(args.data_path_test+' -- no find')

    global path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(args.output_path, date)

    if not os.path.exists(path):
        os.makedirs(path)

    train(args)
