import pathlib
import os
import pandas as pd
import numpy as np
import regex as re
import numpy as np
import random
import cv2


SFT_png = pathlib.Path('oringal_tif')
x =150#初始点坐标
y =150#初始点坐标

def sort_by_number_in_filename(filename):#自定义文件排序函数
    # 使用正则表达式从文件名中提取数字
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[7])
    return match_numbers[7]

png_list = [str(path) for path in sorted(SFT_png.glob('*.tif'),key=sort_by_number_in_filename)]
print(png_list)

i = 0
for file in png_list:
	image = cv2.imread(file,1)
	image_2 = image[y:y+224*3,x:x+224*3] #剪裁图像，第一个维度是纵向
	cv2.imwrite("33tif_{}.tif".format(i),image_2)#保存图像
	i=i+1