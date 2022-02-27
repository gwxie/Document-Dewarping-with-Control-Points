'''
2021/2/3

Guowang Xie

'''
import pickle

import torch
from torch.utils import data
from torch.autograd import Variable, Function

import numpy as np

import sys, os, math

import cv2
import time
import re
from multiprocessing import Pool

import random
import scipy.spatial.qhull as qhull

from scipy.optimize import fsolve
from scipy.interpolate import griddata

def adjust_position(x_min, y_min, x_max, y_max, new_shape):
    if (new_shape[0] - (x_max - x_min)) % 2 == 0:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0
    else:
        f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
        f_g_0_1 = f_g_0_0 + 1

    if (new_shape[1] - (y_max - y_min)) % 2 == 0:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0
    else:
        f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
        f_g_1_1 = f_g_1_0 + 1

    # return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
    return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1

def get_matric_edge(matric):
    return np.concatenate((matric[:, 0, :], matric[:, -1, :], matric[0, 1:-1, :], matric[-1, 1:-1, :]), axis=0)


class SaveFlatImage(object):
    def __init__(self, path, date, date_time, _re_date, data_path_validate, data_path_test, batch_size, preproccess=False):
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        self.preproccess = preproccess
        self.data_path_validate =data_path_validate
        self.data_path_test = data_path_test
        self.batch_size = batch_size
        self.scaling_test_perturbed_img_path = '/lustre/home/gwxie/data/unwarp_new/test/shrink_2048_1920/crop/'
        # self.perturbed_test_img_path = '/lustre/home/gwxie/data/unwarp_new/test/new_1024_960/crop/'
        # self.perturbed_test_img_path = '/lustre/home/gwxie/data/unwarp_new/test/shrink_1024_960/crop/'
        self.perturbed_test_img_path = '/lustre/home/gwxie/data/unwarp_new/test/yin2/'

    def location_mark(self, img, location, color=(0, 0, 255)):
        stepSize = 0
        for l in location.astype(np.int64).reshape(-1, 2):
            cv2.circle(img,
                       (l[0] + math.ceil(stepSize / 2), l[1] + math.ceil(stepSize / 2)), 3, color, -1)
        return img

    def flatByRegressWithClassiy_fiducial_v1_RGB_AT_show(self, fiducial_points, segment, im_name, epoch, perturbed_img=None, scheme='validate', is_scaling=False):
        ''''''
        # if (scheme == 'test' or scheme == 'eval') and is_scaling:
        #     pass
        # else:
        if scheme == 'test' or scheme == 'eval':
            perturbed_img_path = self.data_path_test + im_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
            perturbed_img = cv2.resize(perturbed_img, (960, 1024))
        elif scheme == 'validate' and perturbed_img is None:
            RGB_name = im_name.replace('gw', 'png')
            perturbed_img_path = '/lustre/home/gwxie/data/unwarp_new/train/' + self.data_split + '/validate/png/' + RGB_name
            perturbed_img = cv2.imread(perturbed_img_path, flags=cv2.IMREAD_COLOR)
        elif perturbed_img is not None:
            perturbed_img = perturbed_img.transpose(1, 2, 0)

        fiducial_points = fiducial_points / [992, 992] * [960, 1024]
        # fiducial_points = fiducial_points / [496, 496] * [960, 1024]
        # flat_shape = perturbed_img.shape[:2]
        '''
        tps = cv2.createThinPlateSplineShapeTransformer()
        edge_padding = 3'''
        col_gap = 2 #4
        row_gap = col_gap# col_gap + 1 if col_gap < 6 else col_gap
        # fiducial_point_gaps = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]  # POINTS NUM: 61, 31, 21, 16, 13, 11, 7, 6, 5, 4, 3, 2
        fiducial_point_gaps = [1, 2, 3, 5, 6, 10, 15, 30]        # POINTS NUM: 31, 16, 11, 7, 6, 4, 3, 2
        sshape = fiducial_points[::fiducial_point_gaps[row_gap], ::fiducial_point_gaps[col_gap], :]
        segment_h, segment_w = segment * [fiducial_point_gaps[col_gap], fiducial_point_gaps[row_gap]]
        fiducial_points_row, fiducial_points_col = sshape.shape[:2]
        '''
        im_hight = np.linspace(0, segment_h * (fiducial_points_col - 1), fiducial_points_col, dtype=np.int64)
        im_wide = np.linspace(0, segment_w * (fiducial_points_row - 1), fiducial_points_row, dtype=np.int64)
        im_y, im_x = np.meshgrid(im_hight, im_wide)
        tshape = np.stack((im_x, im_y), axis=2)
        '''
        im_x, im_y = np.mgrid[0:(fiducial_points_col - 1):complex(fiducial_points_col),
                     0:(fiducial_points_row - 1):complex(fiducial_points_row)]

        tshape = np.stack((im_x, im_y), axis=2) * [segment_w, segment_h]

        '''
        tshape = get_matric_edge(tshape)
        sshape = get_matric_edge(sshape)
        '''
        tshape = tshape.reshape(-1, 2)
        sshape = sshape.reshape(-1, 2)
        # perturbed_img_mark = self.location_mark(perturbed_img.copy(), fiducial_points, (0, 0, 255))
        # perturbed_img_mark = self.location_mark(perturbed_img.copy(), sshape, (0, 255, 0))

        '''
        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path,
                                                                                         self.date + self.date_time,
                                                                                         str(epoch))
        if scheme == 'test':
            i_path += '/test'
        if not os.path.exists(i_path):
            os.makedirs(i_path)
        im_name = im_name.replace('gw', 'png')
        cv2.imwrite(i_path + '/' + im_name, perturbed_img_mark)
        # return
        '''
        '''
        matches = list()
        for i in range(sshape.shape[0]):
            matches.append(cv2.DMatch(i, i, 0))
        tps.estimateTransformation(tshape.reshape(1, -1, 2), sshape.reshape(1, -1, 2), matches)
        
        shrink_paddig = 0   # 2 * edge_padding
        x_start, x_end, y_start, y_end = shrink_paddig, segment_h * (fiducial_points_col - 1) - shrink_paddig, shrink_paddig, segment_w * (fiducial_points_row - 1) - shrink_paddig
        # flat_img = tps.warpImage(perturbed_img)[0:segment_h * (fiducial_points_col - 1), 0:segment_w * (fiducial_points_row - 1), :]
        flat_img = tps.warpImage(perturbed_img)[x_start:x_end, y_start:y_end, :]
        # flat_img_mark = self.location_mark(flat_img.copy(), tshape, (0, 255, 0))
        '''

        output_shape = (segment_h * (fiducial_points_col - 1), segment_w * (fiducial_points_row - 1))
        grid_x, grid_y = np.mgrid[0:output_shape[0] - 1:complex(output_shape[0]),
                         0:output_shape[1] - 1:complex(output_shape[1])]
        # grid_z = griddata(tshape, sshape, (grid_y, grid_x), method='cubic').astype('float32')
        grid_ = griddata(tshape, sshape, (grid_y, grid_x), method='linear').astype('float32')
        flat_img = cv2.remap(perturbed_img, grid_[:, :, 0], grid_[:, :, 1], cv2.INTER_CUBIC)

        ''''''
        flat_img = flat_img.astype(np.uint8)

        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path,
                                                                                         self.date + self.date_time,
                                                                                         str(epoch))
        ''''''
        if scheme == 'eval':
            img_figure = cv2.cvtColor(flat_img, cv2.COLOR_RGB2GRAY)
            if scheme == 'eval':
                i_path += '/eval'
            if not os.path.exists(i_path):
                os.makedirs(i_path)
            # print(im_name)
            im_name = im_name.replace(' copy.png', '.jpg')
            cv2.imwrite(i_path + '/' + im_name, img_figure)
        else:
            perturbed_img_mark = self.location_mark(perturbed_img.copy(), sshape, (0, 0, 255))

            shrink_paddig = 0   # 2 * edge_padding
            x_start, x_end, y_start, y_end = shrink_paddig, segment_h * (fiducial_points_col - 1) - shrink_paddig, shrink_paddig, segment_w * (fiducial_points_row - 1) - shrink_paddig

            x_ = (perturbed_img_mark.shape[0]-(x_end-x_start))//2
            y_ = (perturbed_img_mark.shape[1]-(y_end-y_start))//2

            flat_img_new = np.zeros_like(perturbed_img_mark)
            flat_img_new[x_:perturbed_img_mark.shape[0] - x_, y_:perturbed_img_mark.shape[1] - y_] = flat_img
            img_figure = np.concatenate(
                (perturbed_img_mark, flat_img_new), axis=1)

            if scheme == 'test':
                i_path += '/test'
            if not os.path.exists(i_path):
                os.makedirs(i_path)

            im_name = im_name.replace('gw', 'png')
            cv2.imwrite(i_path + '/' + im_name, img_figure)
        '''
        # img_figure = cv2.cvtColor(flat_img, cv2.COLOR_RGB2GRAY)
        # if scheme == 'eval':
        i_path += '/eval'
        if not os.path.exists(i_path):
            os.makedirs(i_path)
        # print(im_name)
        im_name = im_name.replace(' copy.png', '.jpg')
        cv2.imwrite(i_path + '/' + im_name, flat_img)
        '''
    def flatByRegressWithClassiy_multiProcessV2(self, pred_fiducial_points, pred_segment, im_name, epoch, process_pool, perturbed_img=None, scheme='validate', is_scaling=False):
        # process_pool = Pool(self.batch_size)
        for i_val_i in range(pred_fiducial_points.shape[0]):
            # self.flatByRegressWithClassiy_fiducial_v1_RGB_AT(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling)
            process_pool.apply_async(func=self.flatByRegressWithClassiy_fiducial_v1_RGB_AT_show,
                                     args=(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling))
            # process_pool.apply_async(func=self.flatByRegressWithClassiy_fiducial_v1_RGB,
            #                          args=(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling))
            # process_pool.apply_async(func=self.flatByRegressWithClassiy_triangular_v2_RGB,
            #                          args=(pred_fiducial_points[i_val_i], pred_segment[i_val_i], im_name[i_val_i], epoch, None if perturbed_img is None else perturbed_img[i_val_i], scheme, is_scaling))
        # process_pool.close()
        # process_pool.join()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, m=1):
        self.val = val
        self.sum += val * m
        self.count += n
        self.avg = self.sum / self.count

class FlatImg(object):
    def __init__(self, args, path, date, date_time, _re_date, model,\
                 reslut_file, n_classes, optimizer, \
                 model_D=None, optimizer_D=None, \
                 loss_fn=None, loss_fn2=None, data_loader=None, data_loader_hdf5=None, dataPackage_loader = None, \
                 data_path=None, data_path_validate=None, data_path_test=None, data_preproccess=True):     #, valloaderSet, v_loaderSet
        self.args = args
        self.path = path
        self.date = date
        self.date_time = date_time
        self._re_date = _re_date
        # self.valloaderSet = valloaderSet
        # self.v_loaderSet = v_loaderSet
        self.model = model
        self.model_D = model_D
        self.reslut_file = reslut_file
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.optimizer_D = optimizer_D
        self.loss_fn = loss_fn
        self.loss_fn2 = loss_fn2
        self.data_loader = data_loader
        self.data_loader_hdf5 = data_loader_hdf5
        self.dataPackage_loader = dataPackage_loader
        self.data_path = data_path
        self.data_path_validate = data_path_validate
        self.data_path_test = data_path_test
        self.data_preproccess = data_preproccess
        self.save_flat_mage = SaveFlatImage(self.path, self.date, self.date_time, self._re_date, self.data_path_validate, self.data_path_test, self.args.batch_size, self.data_preproccess)

        self.validate_loss = AverageMeter()
        self.validate_loss_regress = AverageMeter()
        self.validate_loss_segment = AverageMeter()
        self.lambda_loss = 1
        self.lambda_loss_segment = 1
        self.lambda_loss_a = 1
        self.lambda_loss_b = 1
        self.lambda_loss_c = 1

    def saveDataPackage(self, data_size='640'):

        if not os.path.exists(self.data_path_validate + 'clip' + data_size + '/'):
            os.makedirs(self.data_path_validate + 'clip' + data_size + '/')

        if not os.path.exists(self.data_path_validate + 'label' + data_size + '/'):
            os.makedirs(self.data_path_validate + 'label' + data_size + '/')
        trainloader = self.loadTrainData(data_split=self.data_split, is_shuffle=True)
        begin_train = time.time()
        for i, (images, labels) in enumerate(trainloader):
            with open(self.data_path_validate + 'clip' + data_size + '/' + str(i) + '.im', 'wb') as f:
                pickle_perturbed_im = pickle.dumps(images)
                f.write(pickle_perturbed_im)

            with open(self.data_path_validate + 'label' + data_size + '/' + str(i) + '.lbl', 'wb') as f:
                pickle_perturbed_lbl = pickle.dumps(labels)
                f.write(pickle_perturbed_lbl)

        trian_t = time.time() - begin_train

        m, s = divmod(trian_t, 60)
        h, m = divmod(m, 60)
        print("All Train Time : %02d:%02d:%02d\n" % (h, m, s))

    def loadTrainData(self, data_split, is_shuffle=True):
        train_loader = self.data_loader(self.data_path, split=data_split, img_shrink=self.args.img_shrink, preproccess=self.data_preproccess)
        trainloader = data.DataLoader(train_loader, batch_size=self.args.batch_size, num_workers=min(self.args.batch_size, 8), drop_last=True, pin_memory=True,
                                      shuffle=is_shuffle)
        return trainloader

    # def loadTrainDataPackage(self, data_split, is_shuffle=True, data_size='640'):
    #     train_loader = self.dataPackage_loader(self.data_path, split=data_split, data_size=data_size)
    #     trainloader = data.DataLoader(train_loader, batch_size=1, num_workers=1, shuffle=is_shuffle)
    #
    #     return trainloader

    def loadValidateAndTestData(self, is_shuffle=True, sub_dir='shrink_512/crop/'):
        v1_loader = self.data_loader(self.data_path_validate, split='validate', img_shrink=self.args.img_shrink, is_return_img_name=True, preproccess=self.data_preproccess)
        valloader1 = data.DataLoader(v1_loader, batch_size=self.args.batch_size, num_workers=min(self.args.batch_size, 8), pin_memory=True, \
                                       shuffle=is_shuffle)

        '''val sets'''
        v_loaderSet = {
            'v1_loader': v1_loader,
        }
        valloaderSet = {
            'valloader1': valloader1,
        }
        # sub_dir = 'crop/crop/'

        t1_loader = self.data_loader(self.data_path_test, split='test', img_shrink=self.args.img_shrink, is_return_img_name=True)
        testloader1 = data.DataLoader(t1_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, pin_memory=True, \
                                       shuffle=False)

        '''test sets'''
        t_loaderSet = {
            't1_loader': v1_loader,
        }
        testloaderSet = {
            'testloader1': testloader1,
        }

        self.valloaderSet = valloaderSet
        self.v_loaderSet = v_loaderSet

        self.testloaderSet = testloaderSet
        self.t_loaderSet = t_loaderSet
        # return v_loaderSet, valloaderSet

    def loadTestData(self, is_shuffle=True):
        t1_loader = self.data_loader(self.data_path_test, split='test', img_shrink=self.args.img_shrink,
                                     is_return_img_name=True)
        testloader1 = data.DataLoader(t1_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size,
                                      pin_memory=True, shuffle=False)

        '''test sets'''
        testloaderSet = {
            'testloader1': testloader1,
        }

        self.testloaderSet = testloaderSet

    def evalData(self, is_shuffle=True, sub_dir='shrink_512/crop/'):
        eval_loader = self.data_loader(self.data_path_test, split='eval', img_shrink=self.args.img_shrink, is_return_img_name=True)
        evalloader = data.DataLoader(eval_loader, batch_size=self.args.batch_size, num_workers=self.args.batch_size, pin_memory=True, \
                                       shuffle=False)

        self.evalloaderSet = evalloader
        # return v_loaderSet, valloaderSet

    def saveModel_epoch(self, epoch):
        epoch += 1
        state = {'epoch': epoch,
                 'model_state': self.model.state_dict(),
                 'optimizer_state': self.optimizer.state_dict(),    # AN ERROR HAS OCCURED
                 }
        i_path = os.path.join(self.path, self.date + self.date_time + ' @' + self._re_date,
                              str(epoch)) if self._re_date is not None else os.path.join(self.path, self.date + self.date_time, str(epoch))
        if not os.path.exists(i_path):
            os.makedirs(i_path)

        if self._re_date is None:
            torch.save(state, i_path + '/' + self.date + self.date_time + "{}".format(self.args.arch) + ".pkl")  # "./trained_model/{}_{}_best_model.pkl"
        else:
            torch.save(state,
                       i_path + '/' + self._re_date + "@" + self.date + self.date_time + "{}".format(
                           self.args.arch) + ".pkl")

    def evalModelGreyC1(self, epoch, is_scaling=False):
        process_pool = Pool(self.args.batch_size*4)

        begin_test = time.time()
        with torch.no_grad():
            # for i_val, (images, perturbed_img, im_name) in enumerate(self.evalloaderSet):
            for i_val, (images, im_name) in enumerate(self.evalloaderSet):
                try:
                    images = Variable(images)

                    outputs, outputs_segment = self.model(images)
                    # outputs, outputs_segment = self.model(images, is_softmax=True)

                    pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                    pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                    self.save_flat_mage.flatByRegressWithClassiy_multiProcess_eval(pred_regress,
                                                                              pred_segment, im_name,
                                                                              epoch + 1, process_pool,
                                                                              # perturbed_img=perturbed_img,
                                                                              scheme='eval',
                                                                              is_scaling=is_scaling)
                except:
                    print('* save image tested error :' + im_name[0])
        process_pool.close()
        process_pool.join()
        test_time = time.time() - begin_test

        print('test time : {test_time:.3f}'.format(
            test_time=test_time))

        print('test time : {test_time:.3f}'.format(
            test_time=test_time),
            file=self.reslut_file)

    def validateOrTestModelV3(self, epoch, trian_t, validate_test='v_l2', is_scaling=False):
        process_pool = Pool(16)# Pool(self.args.batch_size)

        if validate_test == 'v_l4':
            loss_segment_list = 0
            loss_overall_list = 0
            loss_local_list = 0
            loss_edge_list = 0
            loss_rectangles_list = 0
            loss_list = []

            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.valloaderSet.values()):
                    for i_val, (images, labels, segment, im_name) in enumerate(valloader):
                        try:
                            # save_img_ = random.choices([True, False], weights=[1, 0])[0]
                            save_img_ = random.choices([True, False], weights=[0.05, 0.95])[0]
                            # save_img_ = True

                            images = Variable(images)
                            labels = Variable(labels.cuda(self.args.gpu))
                            segment = Variable(segment.cuda(self.args.gpu))

                            outputs, outputs_segment = self.model(images)

                            loss_overall, loss_local, loss_edge, loss_rectangles = self.loss_fn(outputs, labels, size_average=True)
                            loss_segment = self.loss_fn2(outputs_segment, segment)

                            loss = self.lambda_loss * (loss_overall + loss_local + loss_edge * self.lambda_loss_a + loss_rectangles * self.lambda_loss_b) + self.lambda_loss_segment * loss_segment
                            # loss = self.lambda_loss * (loss_local + loss_rectangles + loss_edge*self.lambda_loss_a + loss_overall*self.lambda_loss_b) + self.lambda_loss_segment * loss_segment

                            pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)         # (4, 1280, 1024, 2)
                            pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                            if save_img_:
                                self.save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                                          pred_segment, im_name,
                                                                                          epoch + 1, process_pool,
                                                                                          perturbed_img=images.numpy(), scheme='validate', is_scaling=is_scaling)
                            loss_list.append(loss.item())
                            loss_segment_list += loss_segment.item()
                            loss_overall_list += loss_overall.item()
                            loss_local_list += loss_local.item()
                            # loss_edge_list += loss_edge.item()
                            # loss_rectangles_list += loss_rectangles.item()

                        except:
                            print('* save image validated error :'+im_name[0])
                process_pool.close()
                process_pool.join()
                test_time = time.time() - begin_test

                # if always_save_model:
                #     self.saveModel(epoch, save_path=self.path)
                list_len = len(loss_list)
                print('train time : {trian_t:.3f}\t'
                      'validate time : {test_time:.3f}\t'
                      '[o:{overall_avg:.4f} l:{local_avg:.4f} e:{edge_avg:.4f} r:{rectangles_avg:.4f}\t'
                      '[{loss_regress:.4f}  {loss_segment:.4f}]\n'.format(
                       trian_t=trian_t, test_time=test_time,
                       overall_avg=loss_overall_list / list_len, local_avg=loss_local_list / list_len, edge_avg=loss_edge_list / list_len, rectangles_avg=loss_rectangles_list / list_len,
                       loss_regress=(loss_overall_list+loss_local_list+loss_edge_list) / list_len, loss_segment=loss_segment_list / list_len))
                print('train time : {trian_t:.3f}\t'
                      'validate time : {test_time:.3f}\t'
                      '[o:{overall_avg:.4f} l:{local_avg:.4f} e:{edge_avg:.4f} r:{rectangles_avg:.4f}\t'
                      '[{loss_regress:.4f}  {loss_segment:.4f}]\n'.format(
                       trian_t=trian_t, test_time=test_time,
                       overall_avg=loss_overall_list / list_len, local_avg=loss_local_list / list_len, edge_avg=loss_edge_list / list_len, rectangles_avg=loss_rectangles_list / list_len,
                       loss_regress=(loss_overall_list+loss_local_list+loss_edge_list) / list_len, loss_segment=loss_segment_list / list_len), file=self.reslut_file)
        elif validate_test == 't_all':
            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                    for i_val, (images, im_name) in enumerate(valloader):
                        try:
                            # save_img_ = True
                            save_img_ = random.choices([True, False], weights=[1, 0])[0]
                            # save_img_ = random.choices([True, False], weights=[0.2, 0.8])[0]

                            if save_img_:
                                images = Variable(images)

                                outputs, outputs_segment = self.model(images)
                                # outputs, outputs_segment = self.model(images, is_softmax=True)

                                pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                                pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                                self.save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                                          pred_segment, im_name,
                                                                                          epoch + 1, process_pool,
                                                                                          scheme='test', is_scaling=is_scaling)
                        except:
                            print('* save image tested error :' + im_name[0])
                process_pool.close()
                process_pool.join()
                test_time = time.time() - begin_test

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time))

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time),
                    file=self.reslut_file)
        else:
            begin_test = time.time()
            with torch.no_grad():
                for i_valloader, valloader in enumerate(self.testloaderSet.values()):

                    for i_val, (images, im_name) in enumerate(valloader):
                        try:
                            # save_img_ = True
                            # save_img_ = random.choices([True, False], weights=[1, 0])[0]
                            save_img_ = random.choices([True, False], weights=[0.4, 0.6])[0]

                            if save_img_:
                                images = Variable(images)

                                outputs, outputs_segment = self.model(images)
                                # outputs, outputs_segment = self.model(images, is_softmax=True)

                                pred_regress = outputs.data.cpu().numpy().transpose(0, 2, 3, 1)
                                pred_segment = outputs_segment.data.round().int().cpu().numpy()  # (4, 1280, 1024)  ==outputs.data.argmax(dim=0).cpu().numpy()

                                self.save_flat_mage.flatByRegressWithClassiy_multiProcessV2(pred_regress,
                                                                                          pred_segment, im_name,
                                                                                          epoch + 1, process_pool,
                                                                                          scheme='test', is_scaling=is_scaling)
                        except:
                            print('* save image tested error :' + im_name[0])
                process_pool.close()
                process_pool.join()
                test_time = time.time() - begin_test

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time))

                print('test time : {test_time:.3f}'.format(
                    test_time=test_time),
                    file=self.reslut_file)


