'''
2021/2/3

Guowang Xie

args:
    n_epoch:epoch values for training
    optimizer:various optimization algorithms
    l_rate:initial learning rate
    resume:the path of trained model parameter after
    data_path_train:datasets path for training
    data_path_validate:datasets path for validating
    data_path_test:datasets path for testing
    output-path:output path
    batch_size:
    schema:test or train
    parallel:number of gpus used, like 0, or, 0123

'''
import os, sys
import argparse
import torch
from torch.autograd import Variable
import warnings
import time
import re
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

# import utilsV3 as utils
import utilsV4 as utils

from dataloader import PerturbedDatastsForFiducialPoints_pickle_color_v2_v2

from loss import Losses

def train(args):
    global _re_date
    if args.resume is not None:
        re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
        _re_date = re_date.search(str(args.resume)).group(0)
        reslut_file = open(path + '/' + date + date_time + ' @' + _re_date + '_' + args.arch + '.log', 'w')
    else:
        _re_date = None
        reslut_file = open(path+'/'+date+date_time+'_'+args.arch+'.log', 'w')

    # Setup Dataloader
    data_path = str(args.data_path_train)+'/'
    data_path_validate = str(args.data_path_validate)+'/'
    data_path_test = str(args.data_path_test)+'/'

    print(args)
    print(args, file=reslut_file)

    n_classes = 2

    model = FiducialPoints(n_classes=n_classes, num_filter=32, architecture=DilatedResnetForFlatByFiducialPointsS2, BatchNorm='BN', in_channels=3)     #

    if args.parallel is not None:
        device_ids = list(map(int, args.parallel))
        args.device = torch.device('cuda:'+str(device_ids[0]))

        # args.gpu = device_ids[0]
        # if args.gpu < 8:
        torch.cuda.set_device(args.device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(args.device)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        warnings.warn('no found gpu')
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
            checkpoint = torch.load(args.resume, map_location=args.device)

            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume.name, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume.name))

    loss_fun_classes = Losses(classify_size_average=True, args_gpu=args.device)
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

    if args.schema == 'train':
        trainloader = FlatImg.loadTrainData(data_split='train', is_shuffle=True)
        FlatImg.loadValidateAndTestData(is_shuffle=True)
        trainloader_len = len(trainloader)

        for epoch in range(epoch_start, args.n_epoch):

            print('* lambda_loss :'+str(FlatImg.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']))
            print('* lambda_loss :'+str(FlatImg.lambda_loss)+'\t'+'learning_rate :'+str(optimizer.param_groups[0]['lr']), file=reslut_file)

            begin_train = time.time()
            loss_segment_list = 0
            loss_l1_list = 0
            loss_local_list = 0
            loss_edge_list = 0
            loss_rectangles_list = 0
            loss_list = []

            model.train()
            for i, (images, labels, segment) in enumerate(trainloader):

                images = Variable(images)
                labels = Variable(labels.cuda(args.device))
                segment = Variable(segment.cuda(args.device))

                optimizer.zero_grad()
                outputs, outputs_segment = FlatImg.model(images, is_softmax=False)

                loss_l1, loss_local, loss_edge, loss_rectangles = loss_fun(outputs, labels, size_average=True)
                loss_segment = loss_fun2(outputs_segment, segment)
                loss = FlatImg.lambda_loss*(loss_l1 + loss_local*FlatImg.lambda_loss_a + loss_edge*FlatImg.lambda_loss_b + loss_rectangles*FlatImg.lambda_loss_c) + FlatImg.lambda_loss_segment*loss_segment

                losses.update(loss.item())
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                loss_segment_list += loss_segment.item()
                loss_l1_list += loss_l1.item()
                loss_local_list += loss_local.item()
                # loss_edge_list += loss_edge.item()
                # loss_rectangles_list += loss_rectangles.item()

                if (i + 1) % args.print_freq == 0 or (i + 1) == trainloader_len:
                    list_len = len(loss_list)
                    print('[{0}][{1}/{2}]\t\t'
                          '[{3:.2f} {4:.4f} {5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f} s:{10:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len, loss_segment_list / list_len,
                        loss=losses))
                    print('[{0}][{1}/{2}]\t\t'
                          '[{3:.2f} {4:.4f} {5:.2f}]\t'
                          '[l1:{6:.4f} l:{7:.4f} e:{8:.4f} r:{9:.4f} s:{10:.4f}]\t'
                          '{loss.avg:.4f}'.format(
                        epoch + 1, i + 1, trainloader_len,
                        min(loss_list), sum(loss_list) / list_len, max(loss_list),
                        loss_l1_list / list_len, loss_local_list / list_len, loss_edge_list / list_len, loss_rectangles_list / list_len, loss_segment_list / list_len,
                        loss=losses), file=reslut_file)

                    del loss_list[:]
                    loss_segment_list = 0
                    loss_l1_list = 0
                    loss_local_list = 0
                    loss_edge_list = 0
                    loss_rectangles_list = 0
            FlatImg.saveModel_epoch(epoch)      # FlatImg.saveModel(epoch, save_path=path)

            model.eval()

            trian_t = time.time()-begin_train
            losses.reset()

            train_time.update(trian_t)

            try:
                FlatImg.validateOrTestModelV3(epoch, trian_t, validate_test='v_l4')
                FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t')
            except:
                print(' Error: validate or test')

            try:
                scheduler.step()
            except:
                pass

            print('\n')
    elif args.schema == 'validate':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t_all')
        exit()
    elif args.schema == 'test':
        epoch = checkpoint['epoch'] if args.resume is not None else 0
        model.eval()
        FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t_all')
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

    parser.add_argument('--print-freq', '-p', default=60, type=int,
                        metavar='N', help='print frequency (default: 10)')  # print frequency

    parser.add_argument('--data_path_train', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/', type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_validate', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/validate/', type=str,
                        help='the path of validate images.')  # validate image path

    parser.add_argument('--data_path_test', default=ROOT / 'data/', type=str, help='the path of test images.')

    parser.add_argument('--output-path', default=ROOT / 'flat/', type=str, help='the path is used to  save output --img or result.') 

    parser.add_argument('--resume', default=ROOT / 'ICDAR2021/2021-02-03 16:15:55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl', type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--batch_size', nargs='?', type=int, default=2,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')       # train  validate

    # parser.set_defaults(resume='./ICDAR2021/2021-02-03 16:15:55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl')

    parser.add_argument('--parallel', default='1', type=list,
                        help='choice the gpu id for parallel ')

    args = parser.parse_args()

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise Exception(args.resume+' -- not exist')

    if args.data_path_test is None:
        raise Exception('-- No test path')
    else:
        if not os.path.exists(args.data_path_test):
            raise Exception(args.data_path_test+' -- no find')

    global path, date, date_time
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    date_time = time.strftime(' %H:%M:%S', time.localtime(time.time()))
    path = os.path.join(args.output_path, date)

    if not os.path.exists(path):
        os.makedirs(path)

    train(args)
