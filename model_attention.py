# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:40:45 2022

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, device_num, k=3, p=1):
		super(ResidualBlock, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
			nn.BatchNorm2d(out_channels),
			attention(out_channels, device_num),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
			nn.BatchNorm2d(out_channels),
			attention(out_channels, device_num),
			nn.BatchNorm2d(out_channels)
		)
	def forward(self, x):
		return x + self.net(x)


class ResidualBlock_old(nn.Module):
	def __init__(self, in_channels, out_channels, k=3, p=1):
		super(ResidualBlock_old, self).__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p),
			nn.BatchNorm2d(out_channels),
			attention_old(out_channels),
			nn.BatchNorm2d(out_channels),
			nn.PReLU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p),
			nn.BatchNorm2d(out_channels),
			attention_old(out_channels),
			nn.BatchNorm2d(out_channels)
		)
	def forward(self, x):
		return x + self.net(x)



class attention(nn.Module):
	def __init__(self, channels, device_num, reduction = 4):
		super(attention, self).__init__()
		self.Max=nn.AdaptiveMaxPool2d(1)
		self.Avg=nn.AdaptiveAvgPool2d((1, device_num))
		self.fc=nn.Sequential(
				nn.Linear(channels,channels//reduction,bias=False),
				nn.ReLU(inplace=True),
				nn.Linear(channels//reduction,channels,bias=False),
				nn.Sigmoid()
				)
	def forward(self, x):
		B,C,_,_=x.size()
		out = self.Avg(x)
		out = self.Max(x).view(B,C)
		out=self.fc(out).view(B,C,1,1)
		return x * out.expand_as(x)

class attention_old(nn.Module):
	def __init__(self, channels, reduction = 4):
		super(attention_old, self).__init__()
		self.Avg=nn.AdaptiveAvgPool2d(1)
		self.fc=nn.Sequential(
				nn.Linear(channels,channels//reduction,bias=False),
				nn.ReLU(inplace=True),
				nn.Linear(channels//reduction,channels,bias=False),
				nn.Sigmoid()
				)
	def forward(self, x):
		B,C,_,_=x.size()
		out = self.Avg(x).view(B,C)
		out=self.fc(out).view(B,C,1,1)
		return x * out.expand_as(x)


class sliding_win(nn.Module):
	def __init__(self, num, stride):
		super(sliding_win, self).__init__()
		self.num=int(num)
		self.win_num=int(num-1)
		self.conv1 = nn.Sequential(
			nn.Conv2d(self.num, self.num, kernel_size=(3,9), padding=(1,4), stride = stride),
			nn.BatchNorm2d(self.num),
			nn.PReLU(),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(self.win_num, self.win_num, kernel_size=(3,9), padding=(1,4), stride = stride),
			nn.BatchNorm2d(self.win_num),
			nn.PReLU(),
		)
	def forward(self, x):
		_, _, _, win_size=x.size()
		win_size=int(win_size/self.num)
		for i in range(self.num):
			loc = locals()
			exec("even_%s=x[:,:,:,%d*int(win_size):%d*int(win_size)+win_size]"%(i,i,i))
		y_even = torch.cat([loc['even_'+str(i)] for i in range(self.num)], dim=1)
		y_even=self.conv1(y_even)
		for i in range(self.win_num):
			loc = locals()
			exec("odd_%s=x[:,:,:,%d*int(win_size)+win_size//2:%d*int(win_size)+3*win_size//2]"%(i,i,i))
		y_odd = torch.cat([loc['odd_'+str(i)] for i in range(self.win_num)], dim=1)
		y_odd=self.conv2(y_odd)
		y=torch.cat([y_even, y_odd], dim=1)
		return y


class MWA_ResNet(nn.Module):
	def __init__(self, device_num, n_residual=4):
		super(MWA_ResNet, self).__init__()
		self.n_residual = n_residual
		self.conv1 = nn.Sequential(
			nn.Conv2d(25, 32, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(32),
			nn.PReLU(),
		)
		self.sliding_win_2 = sliding_win(2, (1,4))#输出3
		self.sliding_win_4 = sliding_win(4, (1,2))#输出7
		self.sliding_win_8 = sliding_win(8, 1)#输出15
			
		for i in range(n_residual):
			self.add_module('residual' + str(i+1), ResidualBlock_old(32, 32))
			
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1)),
			nn.BatchNorm2d(32),
			nn.PReLU(),
		)
		self.Feature_extraction = nn.Sequential(
			nn.AdaptiveAvgPool2d((1,device_num)),
			nn.Conv2d(32, 32, kernel_size=1),
			nn.BatchNorm2d(32),
			nn.PReLU(),
			nn.Conv2d(32,1 , kernel_size=1),
		)

	def forward(self, x):
#		y= self.preprocessing(x)
		y_2 = self.sliding_win_2(x)
		y_4 = self.sliding_win_4(x)
		y_8 = self.sliding_win_8(x)
		y = torch.cat((y_2, y_4, y_8), dim=1)
		y = self.conv1(y)
		cache = y.clone()
		for i in range(self.n_residual):
			y = self.__getattr__('residual' + str(i+1))(y)
			
		y = self.conv2(y)
		y = self.Feature_extraction(y + cache)

		return y




class MWRA_ResNet(nn.Module):
	def __init__(self, device_num, n_residual=4):
		super(MWRA_ResNet, self).__init__()
		self.n_residual = n_residual
		self.conv1 = nn.Sequential(
			nn.Conv2d(25, 32, kernel_size=(3,3), padding=(1,1)),
			nn.BatchNorm2d(32),
			nn.PReLU(),
		)
		self.sliding_win_2 = sliding_win(2, (1,4))#输出3
		self.sliding_win_4 = sliding_win(4, (1,2))#输出7
		self.sliding_win_8 = sliding_win(8, 1)#输出15
			
		for i in range(n_residual):
			self.add_module('residual' + str(i+1), ResidualBlock(32, 32, device_num))
			
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=(1,3), padding=(0,1)),
			nn.BatchNorm2d(32),
			nn.PReLU(),
		)
		self.Feature_extraction = nn.Sequential(
			nn.AdaptiveAvgPool2d((1,device_num)),
			nn.Conv2d(32, 32, kernel_size=1),
			nn.BatchNorm2d(32),
			nn.PReLU(),
			nn.Conv2d(32,1 , kernel_size=1),

		)

	def forward(self, x):
#		y= self.preprocessing(x)
		y_2 = self.sliding_win_2(x)
		y_4 = self.sliding_win_4(x)
		y_8 = self.sliding_win_8(x)
		y = torch.cat((y_2, y_4, y_8), dim=1)
		y = self.conv1(y)
		cache = y.clone()
		for i in range(self.n_residual):
			y = self.__getattr__('residual' + str(i+1))(y)
			
		y = self.conv2(y)
		y = self.Feature_extraction(y + cache)
#		print ('G output size :' + str(y.size()))
		return y
