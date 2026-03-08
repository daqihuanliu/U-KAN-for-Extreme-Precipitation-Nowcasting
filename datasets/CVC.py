import torch
from PIL import Image
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from random import sample
import datasets.transforms as T
from datetime import datetime, timedelta
import netCDF4 as nc


class KvasirDataSet(Dataset):
    def __init__(self, Kvasir_folder, img_size=256, train_mode=False, transform=None): 
        super(KvasirDataSet, self).__init__()
        self.img_size = img_size
        self.train_mode = train_mode
        self.transform = transform

        # Set the indices for training and validation
        if self.train_mode:
            self.istr = 0
            self.iend = 7500
        else:
            self.istr = 7500
            self.iend = 9045
        
        # Load statistics
        with open('./output/statistics.txt', 'r') as file:  
            lines = file.readlines()  
            mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
            self.mean_composite_ref = np.float32(mean_str)  
            self.std_composite_ref = np.float32(std_str) 
        
        # Prepare to read the NetCDF file
        self.dataset_path = '/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_1_shuffled.nc'


    def __getitem__(self, index):
        with nc.Dataset(self.dataset_path, 'r') as ds:        
            # Load only the specific slice of data for the current index
            factor = ds.variables['composite_ref'][self.istr + index, :10, :, :]  # load factors
            target = ds.variables['composite_ref'][self.istr + index, 30:40, :,:]     # load targets
            
            # Normalize the factor data
            factor = (factor - self.mean_composite_ref) / self.std_composite_ref
            target = (target - self.mean_composite_ref) / self.std_composite_ref
            
            # Convert to tensors
            factor_tensor = torch.tensor(factor, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            return factor_tensor, target_tensor

    def __len__(self):
        return self.iend - self.istr

def build_dataset(args):
    train_ds = KvasirDataSet(
        args.Kvasir_path,
        args.img_size,
        train_mode=True,
    )

    valid_ds = KvasirDataSet(
        args.Kvasir_path,
        args.img_size,
        train_mode=False,
    )
    
    '''
    index = 48
    data_, target_ = train_ds[index]
    targetnp = target_
    ##############
    import matplotlib
    matplotlib.use('Agg')  # 设置后端为 Agg
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    collev = ["#FFFFFF", "#0000F6", "#01A0F6", "#00ECEC", \
            "#00C800", "#019000", "#FFFF00", \
              "#E7C000", "#FF9000", "#FF0000", "#D60000", \
              "#C00000", "#FF00F0", "#780084", "#AD90F0"]
    
    cmaps = colors.ListedColormap(collev, 'indexed')
    # cm.register_cmap(name='dbzcmap', data=collev, lut=128)
    clevels = range(0, 76, 5)
    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(15, 15), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})  
    fig.subplots_adjust(right=0.8) 
      
    
    for i in range(30):  
        ax = axes.flat[i]  # 使用 flat 属性将二维数组转换为一维数组，方便索引  
        con = ax.contourf(targetnp[i, :, :], levels=clevels, cmap=cmaps)  
        ax.set_title(f'Layer {i+1}')  # 可选：为每个子图设置标题
      
    cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.6])  # [left, bottom, width, height]  
    cbar = fig.colorbar(con, cax=cbar_ax, orientation='vertical')  
    
      
    # 保存图像  
    plt.tight_layout() 
    plt.savefig("targetnp"+str(index)+".png", format='png')  
    plt.close() 
    
    #import sys 
    #sys.exit()
    
    '''
    return train_ds, valid_ds