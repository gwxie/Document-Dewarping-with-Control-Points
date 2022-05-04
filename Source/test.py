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

import time
import re
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

from network import FiducialPoints, DilatedResnetForFlatByFiducialPointsS2

# import utilsV3 as utils
import utilsV4 as utils

from dataloader import PerturbedDatastsForFiducialPoints_pickle_color_v2_v2


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
    else:
        # exit()
        args.device = torch.device('cpu')
        print('using CPU!')


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
            if args.parallel is not None:
                checkpoint = torch.load(args.resume, map_location=args.device)
                model.load_state_dict(checkpoint['model_state'])
            else:
                checkpoint = torch.load(args.resume, map_location=args.device)
                '''cpu'''
                model_parameter_dick = {}
                for k in checkpoint['model_state']:
                    model_parameter_dick[k.replace('module.', '')] = checkpoint['model_state'][k]
                model.load_state_dict(model_parameter_dick)
            
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume.name, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume.name))

    FlatImg = utils.FlatImg(args=args, path=path, date=date, date_time=date_time, _re_date=_re_date, model=model, \
                            reslut_file=reslut_file, n_classes=n_classes, optimizer=optimizer, \
                            data_loader=PerturbedDatastsForFiducialPoints_pickle_color_v2_v2, \
                            data_path=data_path, data_path_validate=data_path_validate, data_path_test=data_path_test, data_preproccess=False)          # , valloaderSet=valloaderSet, v_loaderSet=v_loaderSet
    ''' load data '''
    FlatImg.loadTestData()

    epoch = checkpoint['epoch'] if args.resume is not None else 0
    model.eval()
    FlatImg.validateOrTestModelV3(epoch, 0, validate_test='t_all')
    exit()

    reslut_file.close()

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

    parser.add_argument('--data_path_train', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/color/', type=str,
                        help='the path of train images.')  # train image path

    parser.add_argument('--data_path_validate', default=ROOT / 'dataset/fiducial1024/fiducial1024_v1/validate/', type=str,
                        help='the path of validate images.')  # validate image path

    parser.add_argument('--data_path_test', default=ROOT / 'data/', type=str, help='the path of test images.')

    parser.add_argument('--output-path', default=ROOT / 'flat/', type=str, help='the path is used to  save output --img or result.') 

    parser.add_argument('--resume', default=ROOT / 'ICDAR2021/2021-02-03 16:15:55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl', type=str, 
                        help='Path to previous saved model to restart from')    

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')#28

    parser.add_argument('--schema', type=str, default='test',
                        help='train or test')       # train  validate

    # parser.set_defaults(resume='./ICDAR2021/2021-02-03 16:15:55/143/2021-02-03 16_15_55flat_img_by_fiducial_points-fiducial1024_v1.pkl')

    parser.add_argument('--parallel', default=None, type=list,
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
