import netCDF4 as nc
from datetime import datetime, timedelta
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from timm.models import create_model
import argparse
import sys
sys.path.append('/home/hxiaoyuan2024/projection/evaluate/') 
from models.UNetKAN_main_region.build_models import UKAN_large, UNet_small

#regression task
@torch.no_grad()
def predictor(model, factor_tensor, device):
    #model.eval()

    factor_tensor = factor_tensor.to(device)
    output = model(factor_tensor)
    #print(str(output.float()))
    #print(factor_tensor.shape)
    output_np = output.cpu().squeeze(0).numpy().astype(np.float32)##改为float
    #print(output_np.shape)
    prediction = output_np#np.transpose(output_np, (1, 2, 0)) 

    return prediction


def run_pred(args, model, weights_path, factor_valid, target_valid, factor, idate):

    # get devices
    device = args.device
    #print("using {} device.".format(device))

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state'])
    model.to(device)


    factor_tensor =torch.tensor(factor_valid).to(torch.float32)   ###3_dim
    target_tensor = torch.tensor(target_valid).type(torch.float32)  ###3_dim
    
    #print(target_tensor[0,0, :, :])
    prediction = predictor(model, factor_tensor, device)
    #反标准化
    with open('../UNetKAN_main_region/output_1/statistics.txt', 'r') as file:  
        lines = file.readlines()  
        #print(lines)
        mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
        mean_composite_ref = float(mean_str)  
        std_composite_ref = float(std_str)
    prediction = prediction * std_composite_ref + mean_composite_ref
    #print(np.mean(np.array(prediction[0,:10, :, :])))
    #print(np.mean(np.array(factor[0,:10, :, :])))
    #prediction[0,:, :, :] = np.array(prediction[0,:, :, :])-(np.mean(np.array(prediction[0,:10, :, :]))-np.mean(np.array(factor[0,:10, :, :])))#均值匹配
    #print(np.mean(np.array(prediction[0,:10, :, :])))
    
    
    
    
    '''
    from scipy.stats import gaussian_kde
    import matplotlib
    matplotlib.use('Agg')  # 设置后端为 Agg
    import matplotlib.pyplot as plt
    
    # 生成两个示例序列数据
    data1 = np.array(factor[0,:10, :, :]).flatten()
    data2 = np.array(prediction[0,:10, :, :]).flatten()
    print(np.min(data1),np.min(data2))
    # 估计两个序列的概率密度
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)
     
    # 创建x轴上的点用于绘图，覆盖两个序列的范围
    x = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), 1000)
    density1 = kde1(x)
    density2 = kde2(x)
     
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
     
    # 绘制第一个序列的PDF图
    ax.plot(x, density1, label='Data 1')
     
    # 绘制第二个序列的PDF图
    ax.plot(x, density2, label='Data 2')
     
    # 设置图形的标题和标签
    ax.set_title('Probability Density Functions (PDFs) of Two Sequences')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
     
    # 添加图例
    ax.legend()
    plt.savefig("pdf"+str(idate)+".png", format='png')  
    ''' 
    
    #'''
    ##
    bins = [0, 15, 30, 40, 50, 75, 1000]
    factor_sample = np.array(factor[0,:10, :, :]).flatten()
    counts, _ = np.histogram(factor_sample, bins=bins)
 
    # 计算总数
    total_count = len(factor_sample)
     
    # 使用列表推导式计算占比
    frequencies = [count / total_count  for count in counts]
     
    # 打印结果，使用zip将labels和percentages配对
    #for label, fre in zip(bins[:-1], frequencies):
        #print(f"{label}: {fre:.2f}%")  
        
    #中位数
    factor_meds = []
    factor_sample_sort = np.sort(factor_sample)
    for i in np.arange(0,len(bins)-1):
        med = np.median(factor_sample_sort[(factor_sample_sort>=bins[i]) & (factor_sample_sort<bins[i+1])])
        factor_meds.append(med)

    factor_meds = np.array(factor_meds)
    
    #print(f"factor_meds: {factor_meds}")
        
        
        
    
    ##
    pre_sample = np.array(prediction[0,:10, :, :]).flatten()
    pre_sample_sorted = np.sort(pre_sample)
    # 给定的频率区间
    frequencies = frequencies
     
    # 计算累积频率
    cumulative_frequencies = np.cumsum(frequencies)
     
    # 由于频率是基于样本总数的比例，我们需要知道样本的总数来计算确切的阈值位置
    sample_size = len(pre_sample_sorted)
    #print(sample_size)
    #print(cumulative_frequencies[:-1])
    # 根据累积频率计算阈值索引（注意：这里使用了向下取整，因为索引必须是整数）
    thresholds_indices = [int(sample_size * cf) for cf in cumulative_frequencies[:-1]]  # 排除最后一个累积频率，因为它将是样本的最大值或稍后的位置
    #print(thresholds_indices)
    # 由于我们排除了最后一个累积频率，我们需要添加样本的最大值作为最后一个阈值（或者可以选择不包括它，取决于你的需求）
    thresholds = [pre_sample_sorted[i-1] for i in thresholds_indices] + [pre_sample_sorted[-1]]
     
    # 打印频率所应阈值
    #print(f"thresholds: {thresholds}")
    
    #中位数
    pre_meds = []
    for i in np.arange(0,len(thresholds)):
        if i == 0:
            med = np.median(pre_sample_sorted[(pre_sample_sorted>=pre_sample_sorted[0]) & (pre_sample_sorted<thresholds[i])])
        else:
            med = np.median(pre_sample_sorted[(pre_sample_sorted>=thresholds[i-1]) & (pre_sample_sorted<thresholds[i])])
        pre_meds.append(med)

    pre_meds = np.array(pre_meds)
    
    #print(f"pre_meds: {pre_meds}")
     
    
    
    ##频率匹配订正!!循环匹配,先匹配低等级的重新分布后再逐级循环匹配!!!!!
    for i in np.arange(0,len(prediction[0,:,0,0])):
        for j in np.arange(0,len(bins)-1):
            
            if j == 0:
                expansion =  factor_meds[j]-pre_meds[j] #中位数增量
                #print(expansion)
                condition1 = prediction[0, i, :, :] >= np.min(prediction[0, i, :, :])
                condition2 = prediction[0, i, :, :] < bins[j + 1]
                mask = condition1 & condition2
                prediction[0, i, :, :][mask] += expansion
            else:
                if thresholds[j] != thresholds[j-1]:
                    expansion = factor_meds[j]-pre_meds[j] #中位数增量
                else:
                    expansion = 0
                mask = (prediction[0, i, :, :] >= bins[j]) & (prediction[0, i, :, :] < bins[j + 1])
                prediction[0, i, :, :][mask] += expansion
                #print(expansion)
    
    
    #print(np.mean(prediction[0, :10, :, :]))
    #print(prediction.shape)
    #'''
    
    
    '''
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
      
    
    for i in range(10):  
        ax = axes.flat[i]  # 使用 flat 属性将二维数组转换为一维数组，方便索引  
        con = ax.contourf(prediction[0,i, :, :], levels=clevels, cmap=cmaps)  
        ax.set_title(f'Layer {i+1}')  # 可选：为每个子图设置标题
      
    cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.6])  # [left, bottom, width, height]  
    cbar = fig.colorbar(con, cax=cbar_ax, orientation='vertical')  
    
      
    # 保存图像  
    plt.tight_layout() 
    plt.savefig("./pic_UKAN/pic_2/prediction_"+str(idate)+".png", format='png')  
    plt.close() 
    
    ##############

    fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(15, 15), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})  
    fig.subplots_adjust(right=0.8) 
      
    
    for i in range(10):  
        ax = axes.flat[i]  # 使用 flat 属性将二维数组转换为一维数组，方便索引  
        con = ax.contourf(target_tensor[0,i, :, :], levels=clevels, cmap=cmaps)  
        ax.set_title(f'Layer {i+1}')  # 可选：为每个子图设置标题
      
    cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.6])  # [left, bottom, width, height]  
    cbar = fig.colorbar(con, cax=cbar_ax, orientation='vertical')  
    
     
    # 保存图像  
    plt.tight_layout() 
    plt.savefig("./pic_UKAN/pic_2/target_"+str(idate)+".png", format='png')  
    plt.close()    
    '''
    return prediction[0,:,:,:]
    
def savenc(final_data,seltime,outpath):
    # 保存最终结果
        output_file_path = outpath
        with nc.Dataset(output_file_path, 'w', format='NETCDF4') as nc_out:
            # 创建维度
            nc_out.createDimension('time', final_data.shape[0])
            nc_out.createDimension('series', final_data.shape[1])
            nc_out.createDimension('lat', final_data.shape[2])
            nc_out.createDimension('lon', final_data.shape[3])
            
            # 创建变量并定义属性
            time_var = nc_out.createVariable('time', 'f8', ('time',))
            series_var = nc_out.createVariable('series', 'i4', ('series',))
            lat_var = nc_out.createVariable('lat', 'f4', ('lat',))
            lon_var = nc_out.createVariable('lon', 'f4', ('lon',))
            
            # 添加维度变量的属性
            time_var.units = 'hours since 1970-01-01 00:00:00'
            time_var.description = 'Time in hours from a reference date'
            
            series_var.units = 'count'
            series_var.description = 'Index of the data series'
            
            lat_var.units = 'degrees_north'
            lat_var.description = 'Latitude of the grid points'
            
            lon_var.units = 'degrees_east'
            lon_var.description = 'Longitude of the grid points'
            

            # 添加维度变量内容
            time_var[:] =  seltime#[(t - datetime(1970, 1, 1)).total_seconds() / 3600 for t in seltime]  # 根据需求设置时间值
            series_var[:] = np.arange(final_data.shape[1])  # 根据需求设置序列值
            lat_var[:] = np.linspace(32.2, 44.95, final_data.shape[2])  # 根据实际的纬度范围设置
            lon_var[:] = np.linspace(110.0, 122.75, final_data.shape[3])  # 根据实际的经度范围设置
            
            final_data[final_data < 0] = 0
            # 添加数据
            output_var = nc_out.createVariable('composite_ref', 'B', ('time', 'series', 'lat', 'lon'))
            output_var[:] = final_data[:,:,:,:].astype(np.uint8)
        
        print("outputing finished") 
    
    
##--------------------------------------------------------------------
def get_args_parser():
    parser = argparse.ArgumentParser(
        'UNetKAN training and evaluation script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='UNet_small', type=str, metavar='MODEL',
                        choices=['UKAN_samll', 'UKAN_base', 'UKAN_large','UNet_small'],
                        help='Name of model to train')

    parser.add_argument('--save_weights_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')###cuda:gpu (/cpu)

    return parser




parser = argparse.ArgumentParser(
        'UNetKAN training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

model_pred = create_model(
    args.model,
    args=args
)
print('*******************STARTING PREDICT*******************')
#weights_path = f'/home/hxiaoyuan2024/projection/UNetKAN_main_region/output_3_large/UKAN_large_best_model.pth'
#weights_path = f'/home/hxiaoyuan2024/projection/UNetKAN_main_region/output_1/UKAN_large_best_model.pth'
weights_path = f"/home/hxiaoyuan2024/projection/precip_forecast/UNetKAN_main_region/output_unet2/UNet_small_best_model.pth"
print(weights_path)

with open('../UNetKAN_main_region/output_1/statistics.txt', 'r') as file:  
    lines = file.readlines()  
    #print(lines)
    mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
    mean_composite_ref = float(mean_str)  
    std_composite_ref = float(std_str)

nn = 3525
model_prediction = np.zeros((1348,10,256,256))
target_dataset = np.zeros((1348,10,256,256))
target_composit_ref_timefile_ = np.zeros((1348))
skip = 0
j = 0
for i in np.arange(0,nn):#3525
#for i in np.arange(36,37):#3525
    start = i
    end =  start+2
    with nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r')as dataset:
        factor =  dataset.variables['composite_ref'][start:end,:10,:,:]
        if np.mean(factor[0,:,:,:]) < 3:
            continue
        factor_valid =  (factor - mean_composite_ref)/std_composite_ref
    with nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r')as dataset:
        target_valid = dataset.variables['composite_ref'][start:end,20:30,:]
    with nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r')as dataset:
        target_composit_ref_timefile = dataset.variables['time'][i]
        target_composit_ref_timefile_[j] = target_composit_ref_timefile
        #time_units = target_composit_ref_timefile.units
        #time_values = target_composit_ref_timefile[start+1:end+1+30]
        #start_date = datetime.strptime(time_units.split('since')[-1].strip(), '%Y-%m-%d %H:%M:%S')
        #time_target_composit_ref = [(start_date + timedelta(hours=value)).strftime('%Y-%m-%d %H:%M:%S') for value in time_values]
    #print(time_target_composit_ref)
    #print(factor_valid.shape)
    model_prediction[j,:,:,:] = run_pred(args, model_pred, weights_path, factor_valid, target_valid, factor, start)
    target_dataset[j,:,:,:] = target_valid[0,:,:,:]
    j += 1
del mean_composite_ref,std_composite_ref, model_pred,factor_valid,factor,target_valid,target_composit_ref_timefile

#savenc(target_dataset,target_composit_ref_timefile_,'./output_data/target_output_1.nc')
#del target_dataset
#savenc(model_prediction,target_composit_ref_timefile_,'./output_data/UKAN_output_large_3.nc')
#savenc(model_prediction,target_composit_ref_timefile_,'./output_data/UKAN_output_w_1.nc')
#savenc(model_prediction,target_composit_ref_timefile_,'./output_data/UKAN_output_mse_re_1.nc')
#savenc(model_prediction,target_composit_ref_timefile_,'./output_data/UKAN_output_mse_1.nc')
savenc(model_prediction,target_composit_ref_timefile_,'./output_data/xr_unet_output_mse_2.nc')