import pathlib
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.nn import functional as F
import regex as re
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.models as models
import time




seted_parameter = {'setted_SFT_path':'SFT_log_0_time_shift_0.001_0',
                   'setted_img_transform_height':64,'setted_img_transform_width':64,
                   'setted_batch_size':512,'seted_num_epochs':500, 'seted_lr':0.00001,
                   'setted_loss_and_accuracy_file_name':'Loss_and_accuracy_SFT_log_1_time_shift_0.001_0_Acoustic_signal_detection_model_58_3C','setted_model_path':'58_model_3C',
                   'setted_start_cycle_number':-58,'setted_end_cycle_number':-58,'seted_sample_number/2':4500,
                  }
label_path =  r'optical_signal_processed_label.xls'
if not os.path.exists(seted_parameter['setted_loss_and_accuracy_file_name']):
    os.makedirs(seted_parameter['setted_loss_and_accuracy_file_name'])
if not os.path.exists(seted_parameter['setted_model_path']):
    os.makedirs(seted_parameter['setted_model_path'])


def sort_by_number_in_filename(filename): #Custom file sorting function
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[0]

SFT_png = pathlib.Path(seted_parameter['setted_SFT_path'])
png_list = [str(path) for path in sorted(SFT_png.glob('*.png'),key=sort_by_number_in_filename)]
#print(png_list)
#for ele in png_list:
#	print(ele)


label_file = pd.read_excel(label_path)
label_file = np.array(label_file)
label_file = torch.from_numpy(label_file)
label_data = torch.tensor(label_file)
#print(f'label_shape:',label_data.shape)
#print(label_data)

img_height, img_width = seted_parameter['setted_img_transform_height'], seted_parameter['setted_img_transform_width']
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_height,img_width))
])


class Self_Dataset(Dataset):
	def __init__(self,file_list, labels):
		self.file_list = file_list
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		img = Image.open(self.file_list[index])
		if self.transform is not None:
			img = self.transform(img)[:3,:,:]
		label = self.labels[index]
		return img, label

	def __len__(self):
		return len(self.labels)


#6_DenseNet
class DenseLayer(nn.Module):
    """Basic unit of DenseBlock (DenseLayer) """
    def __init__(self, input_c, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_c, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        # 1×1卷积 bottleneck瓶颈层
        output = self.bn1(inputs)
        output = self.relu1(output)
        output = self.conv1(output)
        # 3×3卷积
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv2(output)

        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate)
        return output

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, input_c, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            concat_features = torch.cat(features, 1)
            # 当前DenseLayer的输出特征
            new_features = layer(concat_features)
            # 收集所有DenseLayer的输出特征
            features.append(new_features)
        return torch.cat(features, 1)

class Transition(nn.Module):
    def __init__(self, input_c, output_c):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(input_c)
        self.relu = nn.ReLU(inplace=True)
        # 1×1卷积
        self.conv = nn.Conv2d(input_c, output_c,
                              kernel_size=1, stride=1, bias=False)
        # 2×2池化
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, input):
        output = self.bn(input)
        output = self.relu(output)
        output = self.conv(output)
        output = self.pool(output)
        return output


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 compression_rate=0.5, drop_rate=0, num_classes=2):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, num_init_features, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),

            #nn.MaxPool2d(3, stride=2, padding=1)
        )


        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               input_c=num_features,
                               bn_size=bn_size,
                               growth_rate=growth_rate,
                               drop_rate=drop_rate)

            self.features.add_module("denseblock%d" % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            
            if i != len(block_config) - 1:
                transition = Transition(input_c=num_features, output_c=int(num_features * compression_rate))
                
                self.features.add_module("transition%d" % (i + 1), transition)
                
                num_features = int(num_features * compression_rate)

        
        self.tail = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        
        self.classifier = nn.Linear(num_features, num_classes)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        tail_output = self.tail(features)
        
        out = F.adaptive_avg_pool2d(tail_output, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def densenet_x(**kwargs):
    return DenseNet(growth_rate=32,
                    block_config=(2, 4, 8, 4),
                    num_init_features=64,
                    bn_size=4,
                    compression_rate=0.5,
                    **kwargs)


#densenet_small = densenet_small()
#a =torch.ones(1,3,64,64)
#b = densenet_small(a)
#print(b)



#Define training function
def train(model, num_epochs, train_d1, valid_d1):
	loss_hist_train = [0] * num_epochs
	accuracy_hist_train = [0] * num_epochs
	loss_hist_valid = [0] * num_epochs
	accuracy_hist_valid = [0] * num_epochs
	TP_list = [0] * num_epochs
	TN_list = [0] * num_epochs
	FP_list = [0] * num_epochs
	FN_list = [0] * num_epochs
	for epoch in range(num_epochs):
		time_start = time.time()
		model.train()
		for x_batch, y_batch in train_d1:
			x_batch = x_batch.to(DEVICE)
			#print(y_batch)
			y_batch = y_batch.to(DEVICE)
			pred = model(x_batch)
            #pred_0, pred_1, pred_2 = model(x_batch)
			y_batch = y_batch.to(torch.long)
			y_batch = y_batch.squeeze()
			#pred = pred.to(torch.float32)
			#print(pred.shape)
			#print(y_batch.shape)
			loss = loss_fn(pred, y_batch)
            #loss_0 = loss_fn(pred_0, y_batch)
			#loss_1 = loss_fn(pred_1, y_batch)
			#loss_2 = loss_fn(pred_2, y_batch)
			#loss = loss_0 + loss_1 + loss_2
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			loss_hist_train[epoch] += loss.item()*y_batch.size(0)
			#print(loss)
			is_correct = (
				torch.argmax(pred, dim=1) == y_batch
			).float()
			accuracy_hist_train[epoch] +=is_correct.sum().cpu()
		loss_hist_train[epoch] /= len(train_d1.dataset)
		accuracy_hist_train[epoch] /= len(train_d1.dataset)
		torch.save(model, os.path.join(seted_parameter['setted_model_path'], 'model_densenet_-58_{}_3C.pt'.format(epoch+1)))

		TP = 0
		TN = 0
		FP = 0
		FN = 0
		model.eval()
		with torch.no_grad():
			for x_batch, y_batch in valid_d1:
				x_batch = x_batch.to(DEVICE)
				y_batch = y_batch.to(DEVICE)
				pred = model(x_batch)
				y_batch = y_batch.to(torch.long)
				y_batch = y_batch.squeeze()
				loss = loss_fn(pred, y_batch)
				loss_hist_valid[epoch] += \
				    loss.item()*y_batch.size(0)
				is_correct =  (
					torch.argmax(pred, dim=1) == y_batch
				).float()


				for index in range(y_batch.shape[0]):
					pre_hunxiao = int(torch.argmax(pred, dim=1)[index].cpu().item())
					label_hunxiao = int(y_batch[index].cpu().item())
					if pre_hunxiao == int(1) and label_hunxiao == int(1):
						TP = TP+1
					elif pre_hunxiao == int(0) and label_hunxiao == int(0):
						TN = TN+1
					elif pre_hunxiao == int(1) and label_hunxiao == int(0):
						FP = FP+1
					else:
						FN = FN+1

				accuracy_hist_valid[epoch] += is_correct.sum().cpu()
			loss_hist_valid[epoch] /= len(valid_d1.dataset)
			accuracy_hist_valid[epoch] /= len(valid_d1.dataset)
			TP_list[epoch] = TP/len(valid_d1.dataset)
			TN_list[epoch] = TN/len(valid_d1.dataset)
			FP_list[epoch] = FP/len(valid_d1.dataset)
			FN_list[epoch] = FN/len(valid_d1.dataset)
			time_end = time.time()
			print(time_end-time_start)

			print(f'Epoch {epoch+1}accuracy: '
				  f'{accuracy_hist_train[epoch]:}.4f val_accuracy: ' 
				  f'{accuracy_hist_valid[epoch]:}')
			#print(TP_list[epoch],TN_list[epoch],FP_list[epoch],FN_list[epoch])
	return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid, TP_list, TN_list, FP_list, FN_list

###############################################################################################
###############################################################################################
######################################Define cycle#############################################
###############################################################################################
###############################################################################################


#set parameter
cyclic_array = [i for i in range(seted_parameter['setted_start_cycle_number'],seted_parameter['setted_end_cycle_number']+1)]
highest_accuracy = np.zeros(len(cyclic_array))
num_epochs = seted_parameter['seted_num_epochs']
#print(cyclic_array)
highest_accuracy_index = 0

for cyclic_number in cyclic_array:
	print(cyclic_number)


	#Process time-lapse and sampling data to establish a dataset
	prcessed_png_list = png_list[1599+cyclic_number:37226+cyclic_number]
	#print(f'prcessed_png_list_len:',len(prcessed_png_list))
	#print(prcessed_png_list)

	processed_label_data = label_data[1599:37226,1]
	#print(2599+cyclic_number)
	label_0_index = []
	label_1_index = []

	number_label_1 = 0
	for index, value in enumerate(processed_label_data):
		if value.numpy() == 1:
			label_1_index.append(index)
			number_label_1=number_label_1+1
			#print(value.numpy())
			#print(index)
		else:
			label_0_index.append(index)
			
	#print(f'number_label_1:',number_label_1)
	#print(len(label_0_index))
	#print(len(label_1_index))
	#selected_label_1_index = label_1_index
	selected_label_1_index = random.sample(label_1_index, seted_parameter['seted_sample_number/2'])
	#selected_label_0_index = random.sample(label_0_index, number_label_1)
	selected_label_0_index = random.sample(label_0_index, seted_parameter['seted_sample_number/2'])
	#print(f'selected_label_0_index:',len(selected_label_0_index))

	#print(processed_label_data)
	#print(torch.zeros([number_label_1]))
	#print(torch.ones([number_label_1]))
	#finnal_label = torch.cat((torch.zeros([number_label_1]),torch.ones([number_label_1])),dim=0)
	finnal_label = torch.cat((torch.zeros([seted_parameter['seted_sample_number/2']]),torch.ones([seted_parameter['seted_sample_number/2']])),dim=0)
	finnal_label = finnal_label.float()
	#print(finnal_label)
	#print(finnal_label)
	selected_label_index = selected_label_0_index+selected_label_1_index
	finnal_png_list = []
	for index in selected_label_index:
		finnal_png_list.append(prcessed_png_list[index])
	#print(f'finnal_png_list:',len(finnal_png_list))

	self_dataset = Self_Dataset(finnal_png_list, finnal_label)
	train_size = int(0.7 * len(self_dataset))  # 训练集大小，占比为70%  
	val_size =  len(self_dataset) - train_size    # 测试集大小，占比为30% 
	train_dataset, valid_dataset = random_split(self_dataset, [train_size, val_size])
	torch.manual_seed(1)
	batch_size = seted_parameter['setted_batch_size']
	train_d1_for_CNN = DataLoader(train_dataset,batch_size,shuffle=True)
	valid_d1_for_CNN = DataLoader(valid_dataset,val_size,shuffle=False)
	
    

    #define model
	densenet_small = densenet_x()
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DEVICE = torch.device('cuda:0')
	densenet_small.to(DEVICE)
	print(torch.cuda.is_available())
	
	#define loss function and optimizer
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(densenet_small.parameters(), lr=seted_parameter['seted_lr'])

	
	#train model
	print(f'cyclic',cyclic_number,f'start')
	torch.manual_seed(1)
	hist = train(densenet_small, num_epochs, train_d1_for_CNN, valid_d1_for_CNN)
	#print(type(hist))

	
	#Draw epoch loss and accuracy curves and save
	plt.clf() 
	x_arr = np.arange(len(hist[0])) + 1
	fig = plt.figure(figsize=(12,4))
	ax = fig.add_subplot(1, 2, 1)
	ax.plot(x_arr, hist[0], '-o', label='Train loss')
	ax.plot(x_arr, hist[1], '--<', label='Validation loss')
	ax.legend(fontsize=15)
	ax.set_xlabel('Epoch', size=15)
	ax.set_ylabel('Loss', size=15)
	ax = fig.add_subplot(1, 2, 2)
	ax.plot(x_arr, hist[2], '-o', label='Train acc.')
	ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
	ax.legend(fontsize=15)
	ax.set_xlabel('Epoch', size=15)
	ax.set_ylabel('Accuracy', size=15)
	plt.savefig(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Loss_and_accuracy_quan_{}.png'.format(cyclic_number)))
	plt.clf() #Clear the content from the plt



	#Save the raw data of the curve
	Train_loss = np.array(hist[0])
	Valid_loss = np.array(hist[1])
	Train_acc = np.array(hist[2])
	Valid_acc = np.array(hist[3])
	TP_list_result = np.array(hist[4])
	TN_list_result = np.array(hist[5])
	FP_list_result = np.array(hist[6])
	FN_list_result = np.array(hist[7])


	hist = np.vstack((x_arr,Train_loss))
	hist = np.vstack((hist,Valid_loss))
	hist = np.vstack((hist,Train_acc))
	hist = np.vstack((hist,Valid_acc))
	hist = np.vstack((hist,TP_list_result))
	hist = np.vstack((hist,TN_list_result))
	hist = np.vstack((hist,FP_list_result))
	hist = np.vstack((hist,FN_list_result))

	hist = np.transpose(hist)
	#print(hist)

	df_hist = pd.DataFrame(hist)
	df_hist.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_loss_and_accuracy_{}.xlsx'.format(cyclic_number)), index=False, header=['Epoch','Train_loss','Valid_loss','Train_acc','Valid_acc','TP','TN','FP','FN'])
	#print(Valid_acc)
	#print(np.max(Valid_acc))
	#print(highest_accuracy)
	#print(highest_accuracy_index)
	highest_accuracy_cycle = np.max(Valid_acc)
	#print(highest_accuracy_cycle)
	highest_accuracy[highest_accuracy_index] = highest_accuracy_cycle #Calculate the highest validation accuracy for each cycle
	#print(highest_accuracy)
	highest_accuracy_index = highest_accuracy_index+1                 #Calculate the index of the highest validation accuracy for the next cycle



#Save the highest validation accuracy for each cycle to Excel
#print(cyclic_array)
#print(highest_accuracy)
highest_accuracy_array = np.vstack((cyclic_array,highest_accuracy))
highest_accuracy_array = np.transpose(highest_accuracy_array)
#print(highest_accuracy_array)
df_highest_accuracy = pd.DataFrame(highest_accuracy_array)
df_highest_accuracy.to_excel(os.path.join(seted_parameter['setted_loss_and_accuracy_file_name'], 'Hist_highest_val_accuracy.xlsx'), index=False, header=['Timeshift','Highest_accuracy'])
