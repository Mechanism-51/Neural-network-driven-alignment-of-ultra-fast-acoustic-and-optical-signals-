import pathlib
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from multiprocessing import Process
import xlrd
import xlwt  
#from xlutils.copy import copyasf 
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import revised_thinkdsp
import matplotlib
import csv  
import re


setted_parameter = {
                     'cycle_start_number':int(0),'cycle_interval':int(5),'cycle_end_number':int(24),'time_shift_length':0.001,
                     'sorted_number_location':int(4),'origin_data_path':'origin_data',
                     'cutted_signal_time':0.025, 'sampling_frequency':int(2000000),'remove_offset':int(1),'log_process':int(0),'normalization':int(1),
                     'seted_seg_length':2382, 'selected_highest_requency':34000
                   }





def read_csv_from_wave_data(file):  
    with open(file, 'r') as f:  
        reader = csv.reader(f, delimiter='\t')  
        origin_data = [row for row in reader]           
        del origin_data[:12]
        data_output = [row[0].rstrip(',').strip().split(',') for row in origin_data]       
        data_output = [[float(num.strip()) for num in row] for row in data_output]
        data_output = np.array(data_output)
        #print(data_output.dtype)
    return data_output



def sort_by_number_in_filename(filename):
    match_numbers = re.findall(r'\d+', os.path.basename(filename))
    match_numbers = [int(num) for num in match_numbers]
    #print(match_numbers[4])
    return match_numbers[setted_parameter['sorted_number_location']]


def exists(variable):
    return variable is not None


def signal_cutting_and_output_SFT_cycle( cycle_start_number,cycle_interval,cycle_end_number,time_shift_length,
                                         sorted_number_location,origin_data_path,
                                         cutted_signal_time,sampling_frequency,
                                         remove_offset,log_process,normalization,seted_seg_length,selected_highest_requency):
    
    saved_file_list = [f"SFT_log_{log_process}_time_shift_{time_shift_length}_{i}" for i in range(cycle_start_number, cycle_end_number+1,cycle_interval)]
    csv_file_0 = glob.glob(os.path.join(origin_data_path,'*.csv'))

    for saved_file in saved_file_list:
        print(f'#########################################################')
        print(saved_file,f'start')
        print(f'#########################################################')
        number = 0
        if not os.path.exists(saved_file):
            os.makedirs(saved_file)
        cutted_signal_point_number = cutted_signal_time*sampling_frequency
        remain_data = np.zeros((int(cutted_signal_point_number+1),2))
        for file_0 in sorted(csv_file_0, key = sort_by_number_in_filename):   
            origin_data = read_csv_from_wave_data(file_0)
            print(file_0,f'start')
            if remain_data.shape[0] < int(cutted_signal_point_number):
                data = np.vstack((remain_data, origin_data))
            else:
                data = origin_data

            parts = saved_file.split('_')
            #print(parts)
            removal_point_number = int(parts[-1])*time_shift_length*sampling_frequency
           
            parts_file_0 = file_0.split('_')
            #print(f'parts_file_0[-2]',parts_file_0[-2])
            #print(parts_file_0)
            if int(parts_file_0[-2])==int(0):
                cycle_number = (data.shape[0]-removal_point_number)//cutted_signal_point_number
                remain_number = (data.shape[0]-removal_point_number)%cutted_signal_point_number
                remain_data = data[int(cutted_signal_point_number*cycle_number+removal_point_number):,:]

            else:
                cycle_number = data.shape[0]//cutted_signal_point_number
                remain_number = data.shape[0]%cutted_signal_point_number
                remain_data = data[int(cutted_signal_point_number*cycle_number):,:]
            for i in range(int(cycle_number)):
                if int(parts_file_0[-2])==int(0):
                    #print(f'chengong')
                    selected_data = data[int(i*cutted_signal_point_number+removal_point_number):int((i+1)*cutted_signal_point_number+removal_point_number),:]
                else:
                    selected_data = data[int(i*cutted_signal_point_number):int((i+1)*cutted_signal_point_number),:]
                #print(i)
                #print(selected_data.shape)
                ts_0 = np.linspace(0,int(cutted_signal_point_number-1),int(cutted_signal_point_number))
                ts_0 = 1/sampling_frequency*ts_0
                ys_0 = selected_data[:,1].ravel()
                if int(remove_offset)==int(1):
                    ys_0 = ys_0-np.mean(ys_0)
                singnal_0 = revised_thinkdsp.Wave(ys=ys_0, ts=ts_0, framerate=sampling_frequency)
                spectrogram_0 = singnal_0.make_spectrogram(seg_length=seted_seg_length) 
                array_0 = spectrogram_0.plot(high=selected_highest_requency)  
                #print(array_0.shape)

                if int(log_process)==int(1):
                    array_0 = np.log10(array_0)

                if int(normalization)==int(1):
                    array_max = np.max(array_0)
                    array_min = np.min(array_0)
                    array_0 = (array_0 - array_min)/(array_max - array_min)
                matplotlib.image.imsave(os.path.join(saved_file,'P_shipintu_{}.png'.format(number)), array_0)
                number = number+1


signal_cutting_and_output_SFT_cycle(**setted_parameter)
print('finsh')