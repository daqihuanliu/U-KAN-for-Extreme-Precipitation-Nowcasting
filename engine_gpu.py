import torch
import torch.nn as nn
import math
import sys
from torch.nn import functional as F
from tqdm import tqdm
from util.metrics import Metrics
import util.utils as utils
import numpy as np
from util.losses import dice_loss, build_target
# from util.losses import BCEDiceLoss as criterion

'''
def weighted_mse_loss(outputs, targets, weights_dict):#先加权等级rmse后平均
    
    with open('./output/statistics.txt', 'r') as file:  
        lines = file.readlines()  
        print(lines)
        mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
        mean_composite_ref = float(mean_str)  
        std_composite_ref = float(std_str)
    
    # 确保outputs和targets的形状相同
    assert outputs.shape == targets.shape, "Outputs and targets must have the same shape."
 
    # 初始化总加权MSE为0
    total_weighted_mse = 0.0
    # 初始化有效的样本数为0
    valid_sample_count = 0
 
    # 遍历weights_dict中的每个范围-权重对
    for (range_low, range_high), weight in weights_dict.items():
        # 创建布尔索引掩码，选中在指定范围内的目标值
        mask = (targets >= (range_low-mean_composite_ref)/std_composite_ref) & (targets < (range_high-mean_composite_ref)/std_composite_ref)
        
        # 应用掩码到outputs和targets上，只保留范围内的元素
        masked_outputs = outputs[mask]
        masked_targets = targets[mask]
        
        # 确保至少有一个有效样本
        if masked_outputs.numel() > 0:
            # 计算范围内的MSE
            mse = F.mse_loss(masked_outputs, masked_targets, reduction='mean')
            #print(mse)
            # 累加加权MSE
            total_weighted_mse += weight * mse
 
    return torch.sqrt(total_weighted_mse)
'''

def weighted_mse_loss(outputs, targets, weights_dict):#逐点加权求rmse
    # 读取均值和标准差
    with open('./output/statistics.txt', 'r') as file:  
        lines = file.readlines()  
        mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
        mean_composite_ref = float(mean_str)  
        std_composite_ref = float(std_str)
    
    # 确保形状一致
    assert outputs.shape == targets.shape, "Outputs和targets形状必须相同"
    
    # 初始化权重张量（与targets同形状）
    weights = torch.zeros_like(targets)
    
    # 遍历权重字典，填充权重张量
    for (range_low, range_high), weight in weights_dict.items():
        # 计算标准化后的范围边界
        low_norm = (range_low - mean_composite_ref) / std_composite_ref
        high_norm = (range_high - mean_composite_ref) / std_composite_ref
        
        # 生成布尔掩码，选择目标值在范围内的点
        mask = (targets >= low_norm) & (targets < high_norm)
        weights[mask] = weight  # 应用权重
    
    # 计算加权MSE（逐点加权后取全局平均）
    weighted_mse = (weights * (outputs - targets) ** 2).mean()
    
    # 计算RMSE
    return torch.sqrt(weighted_mse)


###回归任务(无混合精度训练)
def train_one_epoch(model, optimizer, dataloader, epoch, device, print_freq, clip_grad, clip_mode, loss_scaler, writer=None, args=None):
    model.train()
    num_steps = len(dataloader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # 定义分级权重
    #'''
    weights_dict = {
        (0, 20) : 1,
        (20, 30): 2,
        (30, 40): 5,
        (40, 50): 10,
        (50,float('inf')):30
    }
    #'''
    '''
    weights_dict = {
    (0, 15): 0.2,
    (15, 30): 0.2,
    (30, 40): 0.2,
    (40, 50): 0.2,
    (50, 75): 0.2,
    }
    '''
    for idx, (img, lbl) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        img = img.to(device)
        lbl = lbl.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(img)  
        #print(output.shape)
        # Calculate MSE loss
        #print(output.float())
        
        loss = torch.sqrt(F.mse_loss(output.float(), lbl.float()))  # Using MSE loss for regression
        #print("max:"+str(max(output.float().flatten()))+"_"+str(max(lbl.float().flatten())))
        #print(img.float().flatten().max())
        #print(lbl.float().flatten().max())
        #print(output.float().flatten().max())
        #loss = weighted_mse_loss(output.float(), lbl.float(),weights_dict) #+ 0.01 * sum(p.pow(2).sum() for p in model.parameters())
        
        """
        with open('./_output_.txt', 'w') as f:
            f.write(str(output.float()))
        with open('./_lbl_.txt', 'w') as f:
            f.write(str(lbl.float()))
        print(f"Tensor print saved12")
        """
        loss_value = loss.item()
        #print("batch loss is:"+str(loss_value))
        """
        if not torch.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        """

        # Backward pass and optimization
        #loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode, parameters=model.parameters())
        loss.backward()
        optimizer.step()
        
        lr = optimizer.param_groups[0]["lr"]
        #print(str(idx) + " loss: "+str(loss) + " lr: " + str(lr))
        metric_logger.update(loss=loss_value, lr=lr)
    metric_logger.synchronize_between_processes()
   
    return metric_logger.meters["loss"].global_avg, lr

   

 
@torch.no_grad()#禁用梯度计算减小资源开支
def evaluate_weight(args,model, dataloader, device, print_freq, writer=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    rmse_logger = utils.MetricLogger(delimiter="  ")
    total_loss = 0.0
    sample_count = 0
    
    # 定义分级权重
    #'''
    weights_dict = {
        (0, 20) : 1,
        (20, 30): 2,
        (30, 40): 5,
        (40, 50): 10,
        (50,float('inf')):30
    }
    #'''
    '''
    weights_dict = {
    (0, 15): 0.2,
    (15, 30): 0.2,
    (30, 40): 0.2,
    (40, 50): 0.2,
    (50, 75): 0.2,
    }
    '''
    for idx, (inputs, targets) in enumerate(rmse_logger.log_every(dataloader, print_freq, header='Evaluation:')):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 计算模型输出
        outputs = model(inputs)
        
        # 计算加权MSE损失
        loss = torch.sqrt(F.mse_loss(outputs.float(), targets.float()))  # Using MSE loss for regression
        #loss = weighted_mse_loss(outputs, targets, weights_dict)
        #print(loss)
        # 累加加权MSE和样本数（为了计算平均加权RMSE）
        total_loss += loss.item() * inputs.size(0)
        sample_count += inputs.size(0)
        
        if writer and idx % print_freq == 0:
            # 记录当前的加权RMSE（注意：这里记录的是到当前批次的累积值）
            # 若要记录整个数据集的加权RMSE，应在循环结束后记录
            # 由于我们计算的是累加值，所以这里的step参数可能需要根据实际情况调整
            # 或者，您可以选择不记录每个批次的加权RMSE，而只记录最终的平均加权RMSE
            pass  # 这里暂时不记录，因为我们需要计算整个数据集的平均加权RMSE
    
    # 计算整个数据集的平均加权RMSE
    average_loss = total_loss / sample_count
    
    # 如果需要，可以在这里记录平均加权RMSE到TensorBoard
    if writer:
        writer.add_scalar('eval_weighted_rmse', average_loss, sample_count)
    
    # 清理CUDA缓存（可选）
    torch.cuda.empty_cache()
    print("eval_weighted_rmse: "+str(average_loss))
    return average_loss
