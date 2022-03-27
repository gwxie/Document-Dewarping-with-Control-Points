'''

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class createThinPlateSplineShapeTransformer(nn.Module):

    def __init__(self, I_r_size, fiducial_num=[31, 31], device=torch.device('cuda:0')):
        """
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_r_size : (height, width) of the rectified image I_r
            fiducial_num : the number of fiducial_points
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(createThinPlateSplineShapeTransformer, self).__init__()
        self.f_row_num, self.f_col_num = fiducial_num
        self.F = self.f_row_num * self.f_col_num
        self.I_r_size = I_r_size 
        self.device = device
        self.estimateTransformation = estimateTransformation(self.F, self.I_r_size, self.device)

    def forward(self, batch_I, batch_F, shap_new=None):
        build_P_prime = self.estimateTransformation.build_P_prime(batch_F)  
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        
        if shap_new is None:
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        else:
            build_P_prime_reshape = build_P_prime_reshape.transpose(2, 3).transpose(1, 2)
            map = F.interpolate(build_P_prime_reshape, shap_new,  mode='bilinear', align_corners=True)
            map = map.transpose(1, 2).transpose(2, 3)
            batch_I_r = F.grid_sample(batch_I, map, padding_mode='border', align_corners=True)

        return batch_I_r

class estimateTransformation(nn.Module):

    def __init__(self, F, I_r_size, device):
        super(estimateTransformation, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F) 
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.device = device
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C), dtype=torch.float64, device=self.device)) 
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P), dtype=torch.float64, device=self.device))

    def _build_C(self, F):
        im_x, im_y = np.mgrid[-1:1:complex(31), -1:1:complex(31)]
        C = np.stack((im_y,im_x), axis=2).reshape(-1,2)   
        return C  

    def _build_inv_delta_C(self, F, C):
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C ** 2) 
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = 2 * np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        batch_size = batch_C_prime.size(0)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2).double().to(self.device)), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.matmul(self.inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.matmul(self.P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
