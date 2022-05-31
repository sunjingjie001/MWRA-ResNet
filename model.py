import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, k=3, p=1):
		super(ResidualBlock, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=(k,k), padding=(p,p)),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
#			nn.AvgPool2d(kernel_size=(1,2), stride=1),
			nn.Conv2d(out_channels, out_channels, kernel_size=(k,k), padding=(p,p)),
			nn.BatchNorm2d(out_channels)
		)

	def forward(self, x):
		return x + self.net(x)





class ResNet(nn.Module):
	def __init__(self, device_num, n_residual=4):
		super(ResNet, self).__init__()
		self.n_residual = n_residual
		self.conv1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=(3,9), padding=(1,4)),
			nn.BatchNorm2d(32),
			nn.PReLU(),
		)

		for i in range(n_residual):
			self.add_module('residual' + str(i+1), ResidualBlock(32, 32))
			
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1)),
			nn.PReLU(),
		)

		self.Feature_extraction = nn.Sequential(
			nn.AdaptiveAvgPool2d((1,device_num)),
			nn.Conv2d(32, 32, kernel_size=1),
			nn.PReLU(),
			nn.Conv2d(32,1 , kernel_size=1),
		)

	def forward(self, x):
		y = self.conv1(x)
		cache = y.clone()
		for i in range(self.n_residual):
			y = self.__getattr__('residual' + str(i+1))(y)
			
		y = self.conv2(y)
		y = self.Feature_extraction(y + cache)
		return y






