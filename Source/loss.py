import torch
import torch.nn.functional as F


class Losses(object):
    def __init__(self, classify_size_average=True, args_gpu=0):
        self.classify_size_average = classify_size_average
        self.args_gpu = args_gpu
        self.kernel_r = torch.ones(2, 1, 3, 3).cuda(self.args_gpu)
        self.kernel_cross_17 = torch.tensor(
                                [[[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]],
                                 [[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]]]).cuda(self.args_gpu)
        self.kernel = torch.tensor([[[[0, 1., 0],
                                  [1., 1., 1.],
                                  [0, 1., 0]]],
                                  [[[0, 1., 0],
                                  [1., 1., 1.],
                                  [0, 1., 0]]]]).cuda(self.args_gpu)
        # self.kernel = torch.ones(2, 1, 3, 3).cuda(self.args_gpu)
        self.kernel_2_1 = torch.tensor([[[[1.], [-1.]]], [[[1.], [-1.]]]]).cuda(self.args_gpu)
        self.kernel_1_2 = torch.tensor([[[[1., -1.]]], [[[1., -1.]]]]).cuda(self.args_gpu)
        # self.lambda_ = 0.1
        self.lambda_ = 0.5

        self.matrices_2 = torch.full((1024, 960), 2, dtype=torch.float).cuda(self.args_gpu)
        self.matrices_0 = torch.full((1024, 960), 0, dtype=torch.float).cuda(self.args_gpu)

        self.fiducial_point_gaps = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30]
        self.fiducial_point_gaps_v2 = [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

    def line_cross(self, input, target, size_average=False):

        input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
        target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
        input_arget_rectangles_h = F.mse_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
        input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
        target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
        input_arget_rectangles_o = F.mse_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
        loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

        return loss_rectangles

    def loss_line_cross(self, input, target, size_average=False):
        i_t = target - input

        loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5, 2))
        # loss_local = torch.mean(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5))
        # loss_local = torch.mean(torch.pow(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5, 2))

        input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
        target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
        input_arget_rectangles_h = F.mse_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
        input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
        target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
        input_arget_rectangles_o = F.mse_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
        loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

        return loss_local, loss_rectangles

    def loss_line_cross_l1(self, input, target, size_average=False):
        i_t = target - input

        loss_local = torch.mean(torch.abs(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5))
        # loss_local = torch.mean(torch.abs(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5))
        # loss_local = torch.mean(torch.pow(F.conv2d(i_t, self.kernel, padding=1, groups=2) - i_t*5, 2))

        input_rectangles_h = F.conv2d(input, self.kernel_1_2, padding=0, groups=2)
        target_rectangles_h = F.conv2d(target, self.kernel_1_2, padding=0, groups=2)
        input_arget_rectangles_h = F.l1_loss(input_rectangles_h, target_rectangles_h, size_average=size_average)
        input_rectangles_o = F.conv2d(input, self.kernel_2_1, padding=0, groups=2)
        target_rectangles_o = F.conv2d(target, self.kernel_2_1, padding=0, groups=2)
        input_arget_rectangles_o = F.l1_loss(input_rectangles_o, target_rectangles_o, size_average=size_average)
        loss_rectangles = input_arget_rectangles_h + input_arget_rectangles_o

        return loss_local, loss_rectangles


    def loss_fn4_v5(self, input, target, size_average=False):
        n, c, h, w = input.size()

        # n_ = n*c*h*w

        i_t = target - input

        '''one'''
        loss_l1 = F.smooth_l1_loss(input, target, size_average=size_average)

        '''two'''
        loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (1, 1, 1, 1), mode='replicate'), self.kernel, padding=0, groups=2) - i_t*5, 2))

        '''three   --weak'''
        loss_edge_a = F.mse_loss(input[:, :, 0, :], target[:, :, 0, :], size_average=size_average)
        loss_edge_b = F.mse_loss(input[:, :, h-1, :], target[:, :, h-1, :], size_average=size_average)
        loss_edge_c = F.mse_loss(input[:, :, :, 0], target[:, :, :, 0], size_average=size_average)
        loss_edge_d = F.mse_loss(input[:, :, :, w-1], target[:, :, :, w-1], size_average=size_average)
        loss_edge = loss_edge_a+loss_edge_b+loss_edge_c+loss_edge_d

        '''four'''
        loss_rectangle = self.line_cross(input, target, size_average)


        return loss_l1, loss_local, loss_edge, loss_rectangle

    def loss_fn4_v5_r_4(self, input, target, size_average=False):
        i_t = target - input

        '''one'''
        loss_l1 = F.smooth_l1_loss(input, target, size_average=size_average)


        '''two'''
        loss_local = torch.mean(torch.pow(F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='replicate'), self.kernel_cross_17, padding=0, groups=2) - i_t*17, 2))

        return loss_l1, loss_local, 0, 0
        # return loss_l1, loss_local, loss_edge, loss_rectangle

    def loss_fn_l1_loss(self, input, target):
        return F.l1_loss(input, target, size_average=self.classify_size_average)

