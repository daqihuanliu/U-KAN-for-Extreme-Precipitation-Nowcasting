import os
import time
# import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models import UKAN_samll, UKAN_base, UKAN_large


#regression task
@torch.no_grad()
def predictor(model, factor_tensor, target_tensor, device):
    #model.eval()

    factor_tensor = factor_tensor.to(device)
    output = model(factor_tensor)
    #print(str(output.float()))
    print(factor_tensor.shape)
    output_np = output.cpu().squeeze(0).numpy().astype(np.float32)##改为float
    print(output_np.shape)
    prediction = output_np#np.transpose(output_np, (1, 2, 0)) 

    return prediction


def run_pred(args, model, weights_path, factor_valid, target_valid, factor, idate):

    # get devices
    device = args.device
    print("using {} device.".format(device))

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model_state'])
    model.to(device)


    factor_tensor =torch.tensor(factor_valid).to(torch.float32)   ###3_dim
    target_tensor = torch.tensor(target_valid).type(torch.float32)  ###3_dim
    
    print(target_tensor[0,0, :, :])
    prediction = predictor(model, factor_tensor, target_tensor, device)
    #反标准化
    with open('./output/statistics.txt', 'r') as file:  
        lines = file.readlines()  
        print(lines)
        mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
        mean_composite_ref = float(mean_str)  
        std_composite_ref = float(std_str)
    prediction = prediction * std_composite_ref + mean_composite_ref
    
    
    
    print(np.mean(np.array(prediction[0,:10, :, :])))
    print(np.mean(np.array(factor[0,:10, :, :])))
    #prediction = np.array(prediction[0,:, :, :])/(np.mean(np.array(prediction[0,:10, :, :]))/np.mean(np.array(factor[0,:10, :, :])))#分等级#pdf匹配
    
    '''
    ##
    from scipy.stats import gaussian_kde
    from scipy.integrate import quad
     
    factor_sample = np.array(factor[0,:10, :, :]).flatten()
    print(factor_sample.shape)
    
    # 使用高斯核密度估计来估计PDF
    kde = gaussian_kde(factor_sample)
     
    # 降水区间
    intervals = [
        (0, 15),
        (15, 30),
        (30, 40),
        (40, 45),
        (45, 50),
        (50, 55),
        (55, 70),
        (70, np.inf) 
    ]
     
    probabilities = []
     
    def integrand(x, kde_obj):
        return kde_obj(x)
     
    for interval in intervals:
        # 对于每个区间，使用quad进行数值积分
        if interval[1] == np.inf:
            upper_bound = np.percentile(factor_sample, 100)  
            integral, _ = quad(integrand, interval[0], upper_bound, args=(kde,))
            prob = integral
        else:
            integral, _ = quad(integrand, interval[0], interval[1], args=(kde,))
            prob = integral
        
        probabilities.append(prob)
     
    # 打印结果
    for interval, prob in zip(intervals, probabilities):
        print(f"{interval} is {prob:.4f}")

    ##
    from sklearn.neighbors import KernelDensity
    from scipy.interpolate import interp1d
    
    # 示例样本数据
    pre_sample = np.array(prediction[0,:10, :, :]).flatten()[:, np.newaxis]
    
    # 创建KernelDensity对象
    model = KernelDensity(bandwidth=1.0, kernel='gaussian')
    
    # 拟合模型
    model.fit(pre_sample)
    
    # 生成测试点
    x_test = np.linspace(np.min(pre_sample), np.max(pre_sample), 1000)[:, np.newaxis]
    
    # 计算测试点的密度
    log_dens = model.score_samples(x_test)
    dens = np.exp(log_dens)
    
    # 计算累积概率
    cum_prob = np.cumsum(dens) / np.sum(dens)
    
    # 创建插值函数
    f = interp1d(cum_prob, x_test.ravel(), kind='linear')
    
    # 分区间概率
    probs = probabilities
    
    # 计算累积概率
    cum_probs = np.cumsum(probs)
    print(cum_probs)
    # 使用插值函数找到累积概率对应的阈值
    thresholds = f(cum_probs)
    
    # 计算阈值区间
    intervals = []
    for i in range(len(thresholds) - 1):
        interval = (thresholds[i], thresholds[i + 1])
        intervals.append(interval)
    # 添加最后一个区间，上限为正无穷
    intervals.append((thresholds[-1], np.inf))
    print(intervals)   
    '''
    '''
    ##
    bins = [0, 15, 30, 40, 45, 50, 55, 70, 1000]
    factor_sample = np.array(factor[0,:10, :, :]).flatten()
    counts, _ = np.histogram(factor_sample, bins=bins)
 
    # 计算总数
    total_count = len(factor_sample)
     
    # 使用列表推导式计算占比
    frequencies = [count / total_count  for count in counts]
     
    # 打印结果，使用zip将labels和percentages配对
    for label, fre in zip(bins[:-1], frequencies):
        print(f"{label}: {fre:.2f}%")  
        
    #中位数
    factor_meds = []
    factor_sample_sort = np.sort(factor_sample)
    for i in np.arange(0,len(bins)-1):
        med = np.median(factor_sample_sort[(factor_sample_sort>=bins[i]) & (factor_sample_sort<bins[i+1])])
        factor_meds.append(med)

    factor_meds = np.array(factor_meds)
    
    print(factor_meds)
        
        
        
    
    ##
    pre_sample = np.array(prediction[0,:10, :, :]).flatten()
    pre_sample_sorted = np.sort(pre_sample)
    # 给定的频率区间
    frequencies = frequencies
     
    # 计算累积频率
    cumulative_frequencies = np.cumsum(frequencies)
     
    # 由于频率是基于样本总数的比例，我们需要知道样本的总数来计算确切的阈值位置
    sample_size = len(pre_sample_sorted)
    print(sample_size)
    print(cumulative_frequencies[:-1])
    # 根据累积频率计算阈值索引（注意：这里使用了向下取整，因为索引必须是整数）
    thresholds_indices = [int(sample_size * cf) for cf in cumulative_frequencies[:-1]]  # 排除最后一个累积频率，因为它将是样本的最大值或稍后的位置
    print(thresholds_indices)
    # 由于我们排除了最后一个累积频率，我们需要添加样本的最大值作为最后一个阈值（或者可以选择不包括它，取决于你的需求）
    thresholds = [pre_sample_sorted[i-1] for i in thresholds_indices] + [pre_sample_sorted[-1]]
     
    # 打印频率所应阈值
    print(thresholds)
    
    #中位数
    pre_meds = []
    for i in np.arange(0,len(thresholds)):
        if i == 0:
            med = np.median(pre_sample_sorted[(pre_sample_sorted>=pre_sample_sorted[0]) & (pre_sample_sorted<thresholds[i])])
        else:
            med = np.median(pre_sample_sorted[(pre_sample_sorted>=thresholds[i-1]) & (pre_sample_sorted<thresholds[i])])
        pre_meds.append(med)

    pre_meds = np.array(pre_meds)
    
    print(pre_meds)
    '''
    
    
    ##频率匹配订正
    '''
    for i in np.arange(0,len(prediction[0,:,0,0])):
        for j in np.arange(0,len(bins)-1):
            
            if j == 0:
                expansion = ((bins[j+1]+bins[j])/2.0)/((thresholds[j]+pre_sample_sorted[0])/2.0) #膨胀系数
                print(expansion)
                condition1 = prediction[0, i, :, :] >= np.min(prediction[0, i, :, :])
                condition2 = prediction[0, i, :, :] < bins[j + 1]
                mask = condition1 & condition2
                prediction[0, i, :, :][mask] *= expansion
            else:
                if thresholds[j] != thresholds[j-1]:
                    expansion = ((bins[j+1]+bins[j])/2.0)/((thresholds[j]+thresholds[j-1])/2.0) #膨胀系数
                    print(expansion)
                else:
                    expansion = 1.0
                mask = (prediction[0, i, :, :] >= bins[j]) & (prediction[0, i, :, :] < bins[j + 1])
                prediction[0, i, :, :][mask] *= expansion
    '''
    '''
    for i in np.arange(0,len(prediction[0,:,0,0])):
        for j in np.arange(0,len(bins)-1):
            
            if j == 0:
                expansion = ((bins[j+1]+bins[j])/2.0)-((thresholds[j]+pre_sample_sorted[0])/2.0) #平均值增量——>并不是真正的平均值
                print(expansion)
                condition1 = prediction[0, i, :, :] >= np.min(prediction[0, i, :, :])
                condition2 = prediction[0, i, :, :] < bins[j + 1]
                mask = condition1 & condition2
                prediction[0, i, :, :][mask] += expansion
            else:
                if thresholds[j] != thresholds[j-1]:
                    expansion = ((bins[j+1]+bins[j])/2.0)-((thresholds[j]+thresholds[j-1])/2.0) #平均值增量
                    print(bins[j+1],bins[j],(bins[j+1]+bins[j])/2.0)
                    print(thresholds[j],thresholds[j-1],(thresholds[j]+thresholds[j-1])/2.0)
                else:
                    expansion = 0
                mask = (prediction[0, i, :, :] >= bins[j]) & (prediction[0, i, :, :] < bins[j + 1])
                prediction[0, i, :, :][mask] += expansion
                print(expansion)
    '''
    '''
    for i in np.arange(0,len(prediction[0,:,0,0])):
        for j in np.arange(0,len(bins)-1):
            
            if j == 0:
                expansion =  factor_meds[j]-pre_meds[j] #中位数增量
                print(expansion)
                condition1 = prediction[0, i, :, :] >= np.min(prediction[0, i, :, :])
                condition2 = prediction[0, i, :, :] < bins[j + 1]
                mask = condition1 & condition2
                prediction[0, i, :, :][mask] += expansion
            else:
                if thresholds[j] != thresholds[j-1]:
                    expansion = factor_meds[j]-pre_meds[j] #中位数增量
                    print(bins[j+1],bins[j],(bins[j+1]+bins[j])/2.0)
                    print(thresholds[j],thresholds[j-1],(thresholds[j]+thresholds[j-1])/2.0)
                else:
                    expansion = 0
                mask = (prediction[0, i, :, :] >= bins[j]) & (prediction[0, i, :, :] < bins[j + 1])
                prediction[0, i, :, :][mask] += expansion
                print(expansion)
    #'''
    
    
    print(np.mean(prediction[0, :10, :, :]))
    print(prediction.shape)
    
    
    
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
    plt.savefig("prediction"+str(idate)+".png", format='png')  
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
    plt.savefig("target"+str(idate)+".png", format='png')  
    plt.close() 

# if __name__ == '__main__':
#     run_pred()