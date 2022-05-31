import h5py
import numpy as np
import torch
import os
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch._utils_internal
from torchvision.transforms import ToTensor



def Ascension(One_dimensional_numpy):
	out_shape=One_dimensional_numpy.shape[0]//2
	return np.vstack((One_dimensional_numpy[:out_shape],One_dimensional_numpy[out_shape:]))
	

class LoadDataset(Dataset):
	def __init__(self,dataset_dir, device_num, Signal_num, start_num):
		super(LoadDataset, self).__init__()
		self.dataset_dir=h5py.File(dataset_dir,'r')
		self.dataset_data = self.dataset_dir['data']
		self.dataset_label = self.dataset_dir['label'][0]
		self.device_num=device_num
		a=list(range(0,30000))
		self.index_new=[]
		for i in range(0,device_num):
			self.index_new=self.index_new+a[start_num+1000*i:Signal_num+start_num+1000*i]

	def __getitem__(self, index):
		To_Tensor= ToTensor()
		label = int(self.dataset_label[self.index_new[index]])
		data=Ascension(self.dataset_data[self.index_new[index]])
		data=To_Tensor(data.astype(np.float32))
		return data,label
	def __len__(self):
		return len(self.index_new)




def get_grads(net):
	top = 0
	bottom = 0
	#torch.set_printoptions(precision=10)
	#torch.set_printoptions(threshold=50000)
	for name, param in net.named_parameters():
		if param.requires_grad:
			# Hardcoded param name, subject to change of the network
			if name == 'conv1.0.weight':
				top = param.grad.abs().mean()
				#print (name + str(param.grad))
			# Hardcoded param name, subject to change of the network
			if name == 'Feature_extraction.2.weight':
				bottom = param.grad.abs().mean()
				#print (name + str(param.grad))
	return top, bottom