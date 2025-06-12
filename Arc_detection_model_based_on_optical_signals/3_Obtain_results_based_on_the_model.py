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
from tqdm import tqdm
import torchvision.models as models



seted_parameter = {'setted_tif_path':'cutted_tif','setted_background_path':'background',
                   'setted_img_transform_height':224,'setted_img_transform_width':224,
                   'setted_start_1':0,'setted_end_1':467+1,
                   'output_file_name':'234output.xlsx'
                  }




def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[0])
    return match_numbers[1]

TIF_1 = pathlib.Path(seted_parameter['setted_tif_path'])

tif_list_1 = [str(path) for path in sorted(TIF_1.glob('*.tif'),key=sort_by_number_in_filename)]
tif_list_1 = tif_list_1[seted_parameter['setted_start_1']:seted_parameter['setted_end_1']]

#print(tif_list_1)
#print(tif_list_2)
print(f'len_tif_list_1',len(tif_list_1))
#for ele in tif_list_1:
#	print(ele)
#for ele in tif_list_2:
#	print(ele)



img_height, img_width = seted_parameter['setted_img_transform_height'], seted_parameter['setted_img_transform_width']
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_height,img_width))
])

###############################################################################################
##########################################计算背景##############################################
###############################################################################################
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

###########################由于扣除背景新增加的transform########################################
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


tif_label =torch.zeros(len(tif_list_1))
trained_model = torch.load('squeeze_model_9.pt')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model.to(DEVICE)
#print(models)
self_dataset = Self_Dataset(tif_list_1, tif_label)
self_dataset = DataLoader(self_dataset,128,shuffle=False)



count = 0
result = []
count = 0
for tif_image, label in tqdm(self_dataset):
	#print(tif_image.shape)
	#print(label.shape)
	#tif_image = tif_image.unsqueeze(0)
	tif_image = tif_image.to(DEVICE)
	#with torch.no_grad():
	pred = trained_model(tif_image)
	#print(f'logit:',pred)
	pred = torch.argmax(pred, dim=1).cpu().numpy()
	#print(pred)
	#pred = pred[0]
	result.extend(pred)

#print(f'result：',result)
result_1 = [1 if x == 0 else 0 for x in result]
#print(f'result_1：',result_1)

index = range(0,len(result_1))
#print(f'index',index)
result_array = np.vstack((np.array(index),np.array(result_1)))
result_array = np.transpose(result_array)
#print(result_array)

df = pd.DataFrame(result_array, columns=['index', 'result'])
df.to_excel(seted_parameter['output_file_name'], index=False)