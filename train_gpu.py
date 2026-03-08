import os
import re
import torch
import numpy as np
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer

import torch.backends.cudnn as cudnn

from models.build_models import UKAN_samll, UKAN_base, UKAN_large, UNet_small

from datasets import build_dataset

from util import utils

from scheduler import create_scheduler

from engine_gpu import train_one_epoch,evaluate_weight#cpu/gpu

from estimate_model import run_pred

import netCDF4 as nc
from datetime import datetime, timedelta

import gc

gc.collect()
torch.cuda.empty_cache()

import warnings
warnings.simplefilter('ignore')



def get_args_parser():
    parser = argparse.ArgumentParser(
        'UNetKAN training and evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument("--Kvasir_path", type=str, default='./Kvasir_Dataset/kvasir-seg/Kvasir-SEG/',
                        help="path to Kvasir Dataset")
    parser.add_argument('--predict', default=True, type=bool, help='Estimate Your model')
    parser.add_argument("--img_size", type=int, default=256, help="input size")
    parser.add_argument("--ignore_label", type=int, default=255, help="the dataset ignore_label")
    parser.add_argument("--ignore_index", type=int, default=255, help="the dataset ignore_index")
    parser.add_argument('--data_len', default=7500, type=int,
                        help='count of your entire data_set. For example: ImageNet 1281167')#训练集样本数
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number classes of your dataset')

    parser.add_argument('--batch-size', default=16, type=int)#训练集batch大小 8
    parser.add_argument("--val_batch_size", type=int, default=64, help='batch size for validation (default: 1)')#验证集batch大小
    parser.add_argument('--epochs', default=50, type=int)#20
    parser.add_argument("--train_print_freq", type=int, default=100)
    parser.add_argument("--val_print_freq", type=int, default=200)

    # Model parameters
    parser.add_argument('--model', default='UKAN_samll', type=str, metavar='MODEL',
                        choices=['UKAN_samll', 'UKAN_base', 'UKAN_large','UNet_small'],
                        help='Name of model to train') #'UKAN_samll'

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')#0.02
    parser.add_argument('--clip-mode', type=str, default='agc',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.025)')


    # Learning rate schedule parameters
    parser.add_argument('--sched', default='plateau', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')#余弦退火策略进行学习率调度
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--lr-ep', action='store_true', default=True,
                        help='using the epoch-based scheduler')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=3, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-2, metavar='LR',
                        help='warmup learning rate (default: 1e-3)')#2e-4
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                        help='list of decay epoch indices for multistep lr. must be increasing')
    parser.add_argument('--decay-epochs', type=float, default=1, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')#预热轮数
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--eval-metric', type=str, default='valid_loss',
                        help='detected index for plateau (default:valid_loss)')
    parser.add_argument('--patience-epochs', type=int, default=1, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')


    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--freeze_layers', type=bool, default=False, help='freeze layers')
    parser.add_argument('--set_bn_eval', action='store_true', default=False,
                        help='set BN layers to eval mode during finetuning.')


    parser.add_argument('--save_weights_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--writer_output', default='./',
                        help='path where to save SummaryWriter, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')###cuda:gpu (/cpu)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    #parser.add_argument('--distributed', action='store_true',default=True,help='Enable distributed training')###cpu分布式多线程
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')###分布式多线程
    parser.add_argument('--num_workers', default=16, type=int)#预加载数据进程数
    parser.add_argument('--pin-mem', action='store_true',help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')###gpu才需要
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',help='')###gpu才需要
    parser.set_defaults(pin_mem=True)#

    # training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')#gup参加训练模型进程数
    parser.add_argument('--local_rank', default=0, type=int)###gpu才需要
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequency of model saving')
    return parser



def main(args):
    
    #print(args)
    #print(args.distributed)
    print(";;;")
    utils.init_distributed_mode(args)
    print(";;;;;")
    print(args.distributed)
    #'''
    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))###gpu才需要
    #'''
    #writer = None  # cpu才用

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True###gpu才需要

    # start = time.time()
    best_mIoU = 0.0
    best_F1 = 0.0
    best_acc = 0.0
    best_loss=100000
    device = args.device

    results_file = "results{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_set, valid_set = build_dataset(args)
    """
    index = 1
    data_, mask_ = train_set[index]

    mask_np = mask_.numpy()
    np.savetxt('./mask_np.txt', mask_np.flatten(), delimiter=',')
    print(f"Tensor print saved5")
    """
    print(f"Number of samples in train_set: {len(train_set)}")

    print(args.distributed)
    if args.distributed:
        print("distributed")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        sampler_train = DistributedSampler(train_set, num_replicas=utils.get_world_size(), rank=utils.get_rank(), shuffle=True)#已前全局种子打乱取样
        sampler_val = DistributedSampler(valid_set)
    else:
        print("no distributed")
        sampler_train = RandomSampler(train_set)
        sampler_val = torch.utils.data.SequentialSampler(valid_set)
    
    
    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                             drop_last=True,pin_memory=args.pin_mem, sampler=sampler_train)### gpu才有 pin_memory=args.pin_mem,

    valloader = DataLoader(valid_set, batch_size=args.val_batch_size, num_workers=args.num_workers,
                           drop_last=True, pin_memory=args.pin_mem, sampler=sampler_val)### gpu才有 pin_memory=args.pin_mem,

    model = create_model(
        args.model,
        args=args
    )

    print("///")
    print([args.local_rank])
    model = model.to(device) #gpu才有
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module ###gpu才需要

    """
    #只有cpu多核分布式才需要
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    model_without_ddp = model.module
    """
    n_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('\n********ESTABLISH ARCHITECTURE********')
    print(f'Model: {args.model}\nNumber of parameters: {n_parameters}')
    print('**************************************\n')


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint['model']
        # state_dict = model.state_dict()
        for k in list(checkpoint_model.keys()):
            if 'head' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'head' not in name:
                    para.requires_grad_(False)
                else:
                    print('training {}'.format(name))
    print(0)
    #device = torch.device('cpu')###cpu才需要
    model.to(device)

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.save_weights_dir)
    if args.save_weights_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.save_weights_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")

    print(1)
    checkpoint_name = utils.get_pth_file(args.save_weights_dir)

    if args.resume or checkpoint_name:
        args.resume = os.path.join(f'{args.save_weights_dir}/', checkpoint_name)
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model_state'])
        print(msg)
        if not args.eval:  #用于断续
        
            checkpoint_optimizer_state = checkpoint['optimizer_state']  
            for param_group in checkpoint_optimizer_state['param_groups']:  
                param_group['lr'] = 0.01  ####后设!!!!!
        
            #print(checkpoint['optimizer_state'])
            #Sprint("[[[[[[[[[[[[[[[[[[[[[[")
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            for state in optimizer.state.values():  # load parameters to cuda
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda() ###gpu才需要
            
            checkpoint['scheduler_state']['warmup_lr_init'] = 0.01 ####后设!!!!!!
            
            #print(checkpoint['scheduler_state'])
            #print("_________")
            
            lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
           
            #regression task
            best_loss=  checkpoint['best_loss']#40#
            #"""
            
            #if 'scaler' in checkpoint:
                #loss_scaler.load_state_dict(checkpoint['scaler'])

   
    print(f"Start training for {args.epochs} epochs")

    #'''
    for epoch in range(args.epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        print(2)
        mean_loss, lr = train_one_epoch(model, optimizer, trainloader,
                                        epoch, device, args.train_print_freq, args.clip_grad, args.clip_mode,
                                        loss_scaler, writer, args)
        
        
        valid_loss = evaluate_weight(args, model, valloader, device, args.val_print_freq, writer)
        
        lr_scheduler.step(epoch,valid_loss)#运用学习率调度器
        #"""
        #regression task
        if utils.is_main_process():
            with open(results_file, "a") as f:
                train_info = f"[epoch: {epoch}]\n" \
                             f"train_loss: {mean_loss:.4f}\n" \
                             f"valid_loss: {valid_loss:.4f}\n" \
                             f"lr: {lr:.6f}\n"
                #f.write(train_info + val_info + "\n\n")#classification
                f.write(train_info + "\n\n")
        
        
        if valid_loss < best_loss:#+0.5:
            print(f'decreasing loss: from {best_loss} to {valid_loss}!\n')
            best_loss = valid_loss
            print(f'Min loss: {best_loss}\n')
            if utils.is_main_process():
                checkpoint_save = {
                    "model_state": model_without_ddp.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict(),
                    "best_loss": valid_loss,
                    "scaler": loss_scaler.state_dict()
                } # model information in training
                torch.save(checkpoint_save, f'{args.save_weights_dir}/{args.model}_best_model.pth')
                print('******************Save Checkpoint******************')
                print(f'Save weights to {args.save_weights_dir}/{args.model}_best_model.pth\n')
        else:
            print('*********No improving loss, No saving checkpoint*********')
        #if epoch == 1:
        #    break
        
        
    #'''    
    #'''     
        
    if args.predict and utils.is_main_process():
        model_pred = create_model(
            args.model,
            args=args
        )
        print('*******************STARTING PREDICT*******************')
        weights_path = f'./{args.save_weights_dir}/{args.model}_best_model.pth'
        

        with open('./output/statistics.txt', 'r') as file:  
            lines = file.readlines()  
            print(lines)
            mean_str, std_str = [line.strip().split(': ')[-1] for line in lines]  
            mean_composite_ref = float(mean_str)  
            std_composite_ref = float(std_str)
        
        start = 1336 #1524#1335##1335
        end =  start+2
        
        factor =  nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r').variables['composite_ref'][start:end,:10,:,:]
        factor_valid =  (factor - mean_composite_ref)/std_composite_ref
        target_valid = nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r').variables['composite_ref'][start:end,30:40,:]
        target_composit_ref_timefile = nc.Dataset('/data/groups/g1600002/home/hxiaoyuan2024/data_composite_ref/dataset_composite_ref_2.nc', 'r').variables['time']
        time_units = target_composit_ref_timefile.units
        time_values = target_composit_ref_timefile[start+1:end+1+30]
        start_date = datetime.strptime(time_units.split('since')[-1].strip(), '%Y-%m-%d %H:%M:%S')
        time_target_composit_ref = [(start_date + timedelta(hours=value)).strftime('%Y-%m-%d %H:%M:%S') for value in time_values]
        print(time_target_composit_ref)
        print(factor_valid.shape)
        run_pred(args, model_pred, weights_path, factor_valid, target_valid, factor, start)
    #'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'UNetKAN training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    

    if args.save_weights_dir:
        Path(args.save_weights_dir).mkdir(parents=True, exist_ok=True)
    main(args)