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


seted_parameter = {'setted_tif_path_1':'3-2/cutted_tif','setted_tif_path_2':'3-3/cutted_tif',
                   'setted_background_path':'3_background',
                   'setted_start_1':2601,'setted_end_1':38792,'setted_start_2':1499,'setted_end_2':37326,
                   'setted_img_transform_height':224,'setted_img_transform_width':224,
                   'setted_batch_size':128,'seted_num_epochs':100, 'seted_lr':0.000001,
                   'model_and_loss_accuracy_path':'Model_Loss_and_Accuracy'
                  }
label_path_1 =  r'32_guang_processed_label.xls'
label_path_2 =  r'33_guang_processed_label.xls'

if not os.path.exists(seted_parameter['model_and_loss_accuracy_path']):
    os.makedirs(seted_parameter['model_and_loss_accuracy_path'])


def sort_by_number_in_filename(filename):                          #Custom sorting function
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[1]


TIF_1 = pathlib.Path(seted_parameter['setted_tif_path_1'])
TIF_2 = pathlib.Path(seted_parameter['setted_tif_path_2'])
tif_list_1 = [str(path) for path in sorted(TIF_1.glob('*.tif'),key=sort_by_number_in_filename)]
tif_list_2 = [str(path) for path in sorted(TIF_2.glob('*.tif'),key=sort_by_number_in_filename)]
tif_list_1 = tif_list_1[seted_parameter['setted_start_1']:seted_parameter['setted_end_1']]
tif_list_2 = tif_list_2[seted_parameter['setted_start_2']:seted_parameter['setted_end_2']]
#print(tif_list_1)
#print(tif_list_2)
print(f'len_tif_list_1',len(tif_list_1))
print(f'len_tif_list_2',len(tif_list_2))
#for ele in tif_list_1:
#	print(ele)
#for ele in tif_list_2:
#	print(ele)


label_file_1 = pd.read_excel(label_path_1)
label_file_2 = pd.read_excel(label_path_2)
label_file_1 = np.array(label_file_1)
label_file_1 = label_file_1[seted_parameter['setted_start_1']:seted_parameter['setted_end_1'],1]
label_file_2 = np.array(label_file_2)
label_file_2 = label_file_2[seted_parameter['setted_start_2']:seted_parameter['setted_end_2'],1]
print(f'label_shape_1:',label_file_1.shape)
print(f'label_shape_2:',label_file_2.shape)
#print(label_data_1)
#print(label_data_2)
print(label_file_1.shape)


def caculate_and_sample_file_list(label_file,tif_list):
	number_label_1 = np.sum(label_file)
	tif_label_1 = torch.zeros(number_label_1)
	tif_label_0 = torch.ones(number_label_1)
	tif_flielist_1 = []
	tif_flielist_0 = []
	for index, label in enumerate(label_file):
		if label == 1:
			tif_flielist_1.append(tif_list[index])
		else:
			tif_flielist_0.append(tif_list[index])
	tif_flielist_0 = random.sample(tif_flielist_0, number_label_1)
	#print(tif_label_1)
	#print(tif_label_0)
	#print(tif_flielist_1)
	#print(tif_flielist_0)
	#print(f'shape_tif_label_1:',tif_label_1.shape)
	#print(f'shape_tif_label_0:',tif_label_0.shape)
	#print(f'len_tif_filelist_1:',len(tif_flielist_1))
	#print(f'len_tif_filelist_1:',len(tif_flielist_0))

	return tif_label_1, tif_label_0, tif_flielist_1, tif_flielist_0, number_label_1

tif_label_1_1, tif_label_1_0, tif_flielist_1_1, tif_flielist_1_0, number_label_1_1 =caculate_and_sample_file_list(label_file_1,tif_list_1)
tif_label_2_1, tif_label_2_0, tif_flielist_2_1, tif_flielist_2_0, number_label_2_1 =caculate_and_sample_file_list(label_file_2,tif_list_2)
print(f'len_tif_flielist_1_1',len(tif_flielist_1_1))
print(f'len_tif_flielist_1_0',len(tif_flielist_1_0))
print(f'len_tif_flielist_2_1',len(tif_flielist_2_1))
print(f'len_tif_flielist_2_0',len(tif_flielist_2_0))

tif_label = torch.cat([tif_label_1_1,tif_label_1_0,tif_label_2_1,tif_label_2_0],dim=0)
tif_flielist = tif_flielist_1_1+tif_flielist_1_0+tif_flielist_2_1+tif_flielist_2_0
#print(f'shape_tif_label:',tif_label.shape)
#print(f'len_tif_flielist:',len(tif_flielist))
#print(tif_label)
#print(tif_flielist)



img_height, img_width = seted_parameter['setted_img_transform_height'], seted_parameter['setted_img_transform_width']
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_height,img_width))
])


################################################################################################################
########################################## Background calculation ##############################################
################################################################################################################
bg_path = pathlib.Path(seted_parameter['setted_background_path'])
bg_list = [str(path) for path in sorted(bg_path.glob('*.tif'),key=sort_by_number_in_filename)]
print(f'bg_tif_list:',bg_list)


for index, img_path in enumerate(bg_list):
	img = Image.open(img_path)
	print(img)
	img = transform(img)
	if index != 0:
		img_bg = torch.cat([img_bg,img.unsqueeze(0)],dim=0)
	else:
		img_bg =img.unsqueeze(0)

#print(f'img_bg:',img_bg)
print(f'img_bg_shape:',img_bg.shape)

bg_image = torch.mean(img_bg, dim=0)
print(f'bg_image.shape:',bg_image.shape)


transform_final = transforms.Compose([
	transforms.Normalize(mean=0, std=1)
])
##############################################################################################


class Self_Dataset(Dataset):
	def __init__(self,file_list, labels):
		self.file_list = file_list
		self.labels = labels
		self.transform = transform
		self.transform_final = transform_final


	def __getitem__(self, index):
		img = Image.open(self.file_list[index])
		if self.transform is not None:
			img = (self.transform_final(self.transform(img)-bg_image)-torch.mean(self.transform_final(self.transform(img)-bg_image),dim=(1, 2)).view(3, 1, 1))/torch.std(self.transform_final(self.transform(img)-bg_image),dim=(1, 2)).view(3, 1, 1)
		label = self.labels[index]
		return img, label

	def __len__(self):
		return len(self.labels)

self_dataset = Self_Dataset(tif_flielist, tif_label)
train_size = int(0.7 * len(self_dataset))     # The training set accounts for 70% of the total  
val_size =  len(self_dataset) - train_size    # The proportion of the validation\test set is 30%
train_dataset, valid_dataset = random_split(self_dataset, [train_size, val_size])
torch.manual_seed(1)
batch_size = seted_parameter['setted_batch_size']
train_d1_for_CNN = DataLoader(train_dataset,batch_size,shuffle=True)
valid_d1_for_CNN = DataLoader(valid_dataset,val_size,shuffle=False)
#for tif,label in train_d1_for_CNN:
#	print(tif.shape)
#	print(label.shape)



squeezenet1= models.squeezenet1_1(pretrained=False)
#print(squeezenet1)
num_ftrs = squeezenet1.classifier[1].in_channels                                       # Obtain the input feature count of the current last layer
squeezenet1.classifier[1] = nn.Conv2d(num_ftrs, 2, kernel_size=(1, 1), stride=(1, 1))  # Replace with a new fully connected layer with 2 output features
#resnet18 = models.resnet18(pretrained=False)
#num_ftrs = resnet18.fc.in_features                                                    
#resnet18.fc = torch.nn.Linear(num_ftrs, 2)                                            
#a = torch.ones(1,3,224,224)
#print(squeezenet1(a).shape)
#print(densenet)







#Training Definition
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
		start_time = time.time()
		model.train()
		for x_batch, y_batch in train_d1:
			x_batch = x_batch.to(DEVICE)
			#print(y_batch)
			y_batch = y_batch.to(DEVICE)
			pred = model(x_batch)
			#print(pred)
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
		torch.save(model, 'model_squeezenet1_{}.pt'.format(epoch+1))

		model.eval()
		with torch.no_grad():
			for x_batch, y_batch in valid_d1:
				x_batch = x_batch.to(DEVICE)
				y_batch = y_batch.to(DEVICE)
				pred = model(x_batch)
				#print(pred)
				y_batch = y_batch.to(torch.long)
				y_batch = y_batch.squeeze()
				loss = loss_fn(pred, y_batch)
				loss_hist_valid[epoch] += \
				    loss.item()*y_batch.size(0)
				is_correct =  (
					torch.argmax(pred, dim=1) == y_batch
				).float()
				accuracy_hist_valid[epoch] += is_correct.sum().cpu()
								
				TP = 0
				TN = 0
				FP = 0
				FN = 0
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


			loss_hist_valid[epoch] /= len(valid_d1.dataset)
			accuracy_hist_valid[epoch] /= len(valid_d1.dataset)
			TP_list[epoch] = TP/len(valid_d1.dataset)
			TN_list[epoch] = TN/len(valid_d1.dataset)
			FP_list[epoch] = FP/len(valid_d1.dataset)
			FN_list[epoch] = FN/len(valid_d1.dataset)

			print(f'Epoch {epoch+1}accuracy: '
				  f'{accuracy_hist_train[epoch]:}.4f val_accuracy: ' 
				  f'{accuracy_hist_valid[epoch]:}')
			end_time = time.time()
			print(end_time-start_time)
	return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid, TP_list, TN_list, FP_list, FN_list





DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
squeezenet1.to(DEVICE)
print(torch.cuda.is_available())
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(squeezenet1.parameters(), lr=seted_parameter['seted_lr'])
#torch.manual_seed(1)
start_time = time.time()
hist = train(squeezenet1, seted_parameter['seted_num_epochs'], train_d1_for_CNN, valid_d1_for_CNN)
#end_time = time.time()
#print(f'one_epoch_time',end_time-start_time)
#print(type(hist))



#Save result
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
plt.savefig( 'Loss_and_accuracy_quan.png')
plt.clf() 



#Save result
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
df_hist.to_excel('Hist_loss_and_accuracy.xlsx', index=False, header=['Epoch','Train_loss','Valid_loss','Train_acc','Valid_acc','TP','TN','FP','FN'])
