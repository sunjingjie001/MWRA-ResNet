import os
import argparse
import numpy as np

from tqdm import tqdm
from tensorboard_logger import configure, log_value
import tensorboard_logger
tensorboard_logger.clean_default_logger()
import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data
from torch.utils.data import DataLoader
from model_windows import MWA_ResNet
from utils import LoadDataset, get_grads
def main():

	use_tensorboard = True

	parser = argparse.ArgumentParser(description='SRGAN Train')
	parser.add_argument('--device_num', default=30, type=int, help='train date device_num(0~30)')
	parser.add_argument('--Signal_num', default=500, type=int, help='train date Signal_num(1~1000-start_num)')
	parser.add_argument('--start_num', default=0, type=int, help='train date start_num(0~999)')
	parser.add_argument('--val_Signal_num', default=500, type=int, help='val date Signal_num(1~1000-start_num)')
	parser.add_argument('--val_start_num', default=500, type=int, help='val date start_num(0~999)')
	parser.add_argument('--num_epochs', default=5000, type=int, help='training epoch')
	parser.add_argument('--train_set', default='E:/dataset/Train/dataset_training_aug_0hz.h5', type=str, help='train set path')
	parser.add_argument('--check_point', type=int, default=0, help="continue with previous check_point")

	opt = parser.parse_args()
	n_epoch = opt.num_epochs
	check_point = opt.check_point
	device_num=opt.device_num
	check_point_path = 'E:/ResNet/ResNet_交叉熵多尺度_2_4_8_注意力改进/result__'+str(check_point)+'_start'
	if not os.path.exists(check_point_path):
		os.makedirs(check_point_path)

	train_set = LoadDataset(opt.train_set, device_num, opt.Signal_num, opt.start_num)
	train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=1, shuffle=True)
	val_set = LoadDataset(opt.train_set, device_num, opt.val_Signal_num, opt.val_start_num)
	val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=True)









	crossloss = nn.CrossEntropyLoss()
	if not torch.cuda.is_available():
		print ('!!!!!!!!!!!!!!USING CPU!!!!!!!!!!!!!')

	netResNet = MWA_ResNet(device_num)
	print('# ResNet parameters:', sum(param.numel() for param in netResNet.parameters()))


	if torch.cuda.is_available():
		netResNet.cuda()

		crossloss.cuda()
	
	if use_tensorboard:
	
		configure('E:/log对比/ResNet_交叉熵多尺度_2_4_8_注意力改进', flush_secs=5)
	
	
	optimizerResNet = optim.Adam(netResNet.parameters(), lr=1e-5)



	if check_point != 0:
		if torch.cuda.is_available():
			netResNet.load_state_dict(torch.load('E:/ResNet/ResNet_交叉熵多尺度_2_4_8_注意力改进/result__0_start/netResNet_epoch_' + str(check_point) + '_gpu.pth'))

		else :
			netResNet.load_state_dict(torch.load('E:/ResNet/ResNet_交叉熵多尺度_2_4_8_注意力改进/result__0_start/netResNet_epoch_' + str(check_point) + '_cpu.pth'))

		
		
		
		
		
		
		
		
		
	for epoch in range(1 + max(check_point, 0), n_epoch + 1 + max(check_point, 0)):
		train_bar = tqdm(train_loader)

		

		
		netResNet.train()
		
		cache = {'ResNet_loss': 0,'true_predict': 0, 'grads': 0, 'val_ResNet_loss': 0,'val_true_predict': 0, 'true_predict_possibility': 0, 'val_true_predict_possibility': 0}

		for Signal_1, label in train_bar:

			if torch.cuda.is_available():
				Signal_1 = Signal_1.cuda()
				label=label.cuda()
			fake_Signal_1 = netResNet(Signal_1)[0, 0, :, :]

			
			netResNet.zero_grad()

			ResNet_loss =crossloss(fake_Signal_1 , label-1)#
 

			cache['ResNet_loss'] += ResNet_loss.mean().item()

			ResNet_loss.backward()
			optimizerResNet.step()

			gtg, gbg = get_grads(netResNet)
			Signal_label_value=fake_Signal_1[0].softmax(dim=0)
#			print(Signal_label_value.shape)
			predict_label_probability=torch.max(Signal_label_value)
			predict_label=list(torch.where(Signal_label_value==torch.max(Signal_label_value)))[0][0]+1
			true_or_false= predict_label==label
#			print(predict_label,label,true_or_false)
			cache['true_predict'] += true_or_false
			cache['grads'] += gtg
			if predict_label_probability >= 0.9:
				cache['true_predict_possibility'] += true_or_false


			# Print information by tqdm
			train_bar.set_description(desc='[%d/%d] G grads:(%f, %f) Loss_G: %.4f predict_label: %.4f label: %.4f predict_label_probability: %.4f true_or_false: %.4f' % (epoch, n_epoch, gtg, gbg, ResNet_loss, predict_label, label, predict_label_probability, true_or_false))

		if use_tensorboard:
			log_value('ResNet_loss', cache['ResNet_loss']/len(train_loader), epoch)
			log_value('true_predict', int(cache['true_predict'])/len(train_loader), epoch)
			log_value('grads', cache['grads']/len(train_loader), epoch)
			log_value('true_predict_possibility', int(cache['true_predict_possibility'])/len(val_loader), epoch)
		print(cache['true_predict']/len(train_loader))



		val_bar = tqdm(val_loader)

		# this is train for val_test
		for val_Signal_1, val_label in val_bar:
			if torch.cuda.is_available():
				val_Signal_1 = val_Signal_1.cuda()
				val_label = val_label.cuda()
			val_fake_Signal_1 = netResNet(val_Signal_1)[0, 0, :, :]

			val_ResNet_loss =crossloss(val_fake_Signal_1 , val_label-1)
			cache['val_ResNet_loss'] += val_ResNet_loss.mean().item()


			val_Signal_label_value=val_fake_Signal_1[0].softmax(dim=0)
			val_predict_label_probability=torch.max(val_Signal_label_value)
			val_predict_label=list(torch.where(val_Signal_label_value==torch.max(val_Signal_label_value)))[0][0]+1
			val_true_or_false= val_predict_label==val_label
			cache['val_true_predict'] += val_true_or_false
			if val_predict_label_probability >= 0.9:
				cache['val_true_predict_possibility'] += val_true_or_false 

		if use_tensorboard:
			log_value('val_ResNet_loss', cache['val_ResNet_loss']/len(val_loader), epoch)
			log_value('val_true_predict', int(cache['val_true_predict'])/len(val_loader), epoch)
			log_value('val_true_predict_possibility', int(cache['val_true_predict_possibility'])/len(val_loader), epoch)
		
		# Save model parameters	
		if torch.cuda.is_available():
			torch.save(netResNet.state_dict(), check_point_path + '/netResNet_epoch_%d_gpu.pth' % (epoch))
			if epoch%5 == 0:
				torch.save(optimizerResNet.state_dict(), check_point_path + '/optimizerResNet_epoch_%d_gpu.pth' % (epoch))
		else:
			torch.save(netResNet.state_dict(), check_point_path + '/netResNet_epoch_%d_cpu.pth' % (epoch))
			if epoch%5 == 0:
				torch.save(optimizerResNet.state_dict(), check_point_path + '/optimizerResNet_epoch_%d_cpu.pth' % (epoch))
				


if __name__ == '__main__':
	main()
