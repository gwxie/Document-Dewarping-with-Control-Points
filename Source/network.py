import math
import pickle
import torch.nn as nn
import torch
# from xgw.dewarp.fiducial_points.networks.resnet import *
import torch.nn.init as tinit
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,  stride=stride, padding=1)

def dilation_conv_bn_act(in_channels, out_dim, act_fn, BatchNorm, dilation=4):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
		BatchNorm(out_dim),
		# nn.BatchNorm2d(out_dim),
		act_fn,
	)
	return model

def dilation_conv(in_channels, out_dim, stride=1, dilation=4, groups=1):
	model = nn.Sequential(
		nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups),
	)
	return model

class ResidualBlockWithDilatedV1(nn.Module):
	def __init__(self, in_channels, out_channels, BatchNorm, stride=1, downsample=None, is_activation=True, is_top=False, is_dropout=False):
		super(ResidualBlockWithDilatedV1, self).__init__()
		self.stride = stride
		self.is_activation = is_activation
		self.downsample = downsample
		self.is_top = is_top
		if self.stride != 1 or self.is_top:
			self.conv1 = conv3x3(in_channels, out_channels, self.stride)
		else:
			self.conv1 = dilation_conv(in_channels, out_channels, dilation=3)		# 3
		self.bn1 = BatchNorm(out_channels)
		# self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		if self.stride != 1 or self.is_top:
			self.conv2 = conv3x3(out_channels, out_channels)
		else:
			self.conv2 = dilation_conv(out_channels, out_channels, dilation=3)		# 1
		self.bn2 = BatchNorm(out_channels)

		self.is_dropout = is_dropout
		self.drop_out = nn.Dropout2d(p=0.2)

	def forward(self, x):
		residual = x

		out1 = self.relu(self.bn1(self.conv1(x)))
		# if self.is_dropout:
		# 	out1 = self.drop_out(out1)
		out = self.bn2(self.conv2(out1))
		if self.downsample is not None:
			residual = self.downsample(x)
		out += residual
		out = self.relu(out)

		return out

class ResNetV2StraightV2(nn.Module):
	def __init__(self, num_filter, map_num, BatchNorm, block_nums=[3, 4, 6, 3], block=ResidualBlockWithDilatedV1, stride=[1, 2, 2, 2], dropRate=[0.2, 0.2, 0.2, 0.2], is_sub_dropout=False):
		super(ResNetV2StraightV2, self).__init__()
		self.in_channels = num_filter * map_num[0]
		self.dropRate = dropRate
		self.stride = stride
		self.is_sub_dropout = is_sub_dropout
		# self.is_dropout = is_dropout
		self.drop_out = nn.Dropout2d(p=dropRate[0])
		self.drop_out_2 = nn.Dropout2d(p=dropRate[1])
		self.drop_out_3 = nn.Dropout2d(p=dropRate[2])
		self.drop_out_4 = nn.Dropout2d(p=dropRate[3]) 		# add
		self.relu = nn.ReLU(inplace=True)

		self.block_nums = block_nums
		self.layer1 = self.blocklayer(block, num_filter * map_num[0], self.block_nums[0], BatchNorm, stride=self.stride[0])
		self.layer2 = self.blocklayer(block, num_filter * map_num[1], self.block_nums[1], BatchNorm, stride=self.stride[1])
		self.layer3 = self.blocklayer(block, num_filter * map_num[2], self.block_nums[2], BatchNorm, stride=self.stride[2])
		self.layer4 = self.blocklayer(block, num_filter * map_num[3], self.block_nums[3], BatchNorm, stride=self.stride[3])

	def blocklayer(self, block, out_channels, block_nums, BatchNorm, stride=1):
		downsample = None
		if (stride != 1) or (self.in_channels != out_channels):
			downsample = nn.Sequential(
				conv3x3(self.in_channels, out_channels, stride=stride),
				BatchNorm(out_channels))

		layers = []
		layers.append(block(self.in_channels, out_channels, BatchNorm, stride, downsample, is_top=True, is_dropout=False))
		self.in_channels = out_channels
		for i in range(1, block_nums):
			layers.append(block(out_channels, out_channels, BatchNorm, is_activation=True, is_top=False, is_dropout=self.is_sub_dropout))
		return nn.Sequential(*layers)

	def forward(self, x, is_skip=False):

		out1 = self.layer1(x)

		out2 = self.layer2(out1)

		out3 = self.layer3(out2)

		out4 = self.layer4(out3)

		return out4

class FiducialPoints(nn.Module):
	def __init__(self, n_classes, num_filter, architecture, BatchNorm='GN', in_channels=3):
		super(FiducialPoints, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.num_filter = num_filter
		if BatchNorm == 'IN':
			BatchNorm = nn.InstanceNorm2d
		elif BatchNorm == 'BN':
			BatchNorm = nn.BatchNorm2d
		elif BatchNorm == 'GN':
			BatchNorm = nn.GroupNorm



		self.dilated_unet = architecture(self.n_classes, self.num_filter, BatchNorm, in_channels=self.in_channels)

	def forward(self, x, is_softmax=True):
		return self.dilated_unet(x, is_softmax)

''' Dilated Resnet For Flat By Classify with Rgress   simple -2'''
class DilatedResnetForFlatByFiducialPointsS2(nn.Module):

	def __init__(self, n_classes, num_filter, BatchNorm, in_channels=3):
		super(DilatedResnetForFlatByFiducialPointsS2, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.num_filter = num_filter
		# act_fn = nn.PReLU()
		act_fn = nn.ReLU(inplace=True)
		# act_fn = nn.LeakyReLU(0.2)

		map_num = [1, 2, 4, 8, 16]

		print("\n------load DilatedResnetForFlatByFiducialPointsS2------\n")

		self.resnet_head = nn.Sequential(

			nn.Conv2d(self.in_channels, self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1),
			# nn.InstanceNorm2d(self.num_filter * map_num[0]),
			# BatchNorm(1, self.num_filter * map_num[0]),
			BatchNorm(self.num_filter * map_num[0]),
			# nn.BatchNorm2d(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
			# nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
			nn.Conv2d(self.num_filter * map_num[0], self.num_filter * map_num[0], kernel_size=3, stride=2, padding=1),
			BatchNorm(self.num_filter * map_num[0]),
			act_fn,
			# nn.Dropout(p=0.2),
		)

		self.resnet_down = ResNetV2StraightV2(num_filter, map_num, BatchNorm, block_nums=[3, 4, 6, 3], block=ResidualBlockWithDilatedV1, dropRate=[0, 0, 0, 0], is_sub_dropout=False)

		map_num_i = 3
		self.bridge_1 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								 act_fn, BatchNorm, dilation=1),
			# conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i], act_fn),
		)
		self.bridge_2 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_3 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=5),
		)
		self.bridge_4 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=8),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=3),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=2),
		)
		self.bridge_5 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=7),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=4),
		)
		self.bridge_6 = nn.Sequential(
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=18),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=12),
			dilation_conv_bn_act(self.num_filter * map_num[map_num_i], self.num_filter * map_num[map_num_i],
								act_fn, BatchNorm, dilation=6),
		)

		self.bridge_concate = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[map_num_i] * 6, self.num_filter * map_num[2], kernel_size=1, stride=1, padding=0),
			# BatchNorm(GN_num, self.num_filter * map_num[4]),
			BatchNorm(self.num_filter * map_num[2]),
			# nn.BatchNorm2d(self.num_filter * map_num[4]),
			act_fn,
		)
		self.out_regress = nn.Sequential(
			nn.Conv2d(self.num_filter * map_num[2], self.num_filter * map_num[0], kernel_size=3, stride=1, padding=1),
			BatchNorm(self.num_filter * map_num[0]),
			nn.PReLU(),

			nn.Conv2d(self.num_filter * map_num[0], n_classes, kernel_size=3, stride=1, padding=1),

		)


		self.segment_regress = nn.Linear(self.num_filter * map_num[2]*31*31, 2)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				tinit.xavier_normal_(m.weight, gain=0.2)
			if isinstance(m, nn.ConvTranspose2d):
				assert m.kernel_size[0] == m.kernel_size[1]
				tinit.xavier_normal_(m.weight, gain=0.2)

	def cat(self, trans, down):
		return torch.cat([trans, down], dim=1)

	def forward(self, x, is_softmax):
		resnet_head = self.resnet_head(x)
		resnet_down = self.resnet_down(resnet_head)

		bridge_1 = self.bridge_1(resnet_down)
		bridge_2 = self.bridge_2(resnet_down)
		bridge_3 = self.bridge_3(resnet_down)
		bridge_4 = self.bridge_4(resnet_down)
		bridge_5 = self.bridge_5(resnet_down)
		bridge_6 = self.bridge_6(resnet_down)
		bridge_concate = torch.cat([bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], dim=1)
		bridge = self.bridge_concate(bridge_concate)

		out_regress = self.out_regress(bridge)

		segment_regress = self.segment_regress(bridge.view(x.size(0), -1))

		return out_regress, segment_regress
