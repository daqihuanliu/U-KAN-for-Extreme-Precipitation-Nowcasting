<h1 align='center'>U-KAN</h1>

## This is a warehouse for UKAN-Pytorch-model, can be used to train your extreme precipitation dataset for nowcasting.
### The code partly come from [official source code](https://github.com/CUHK-AIM-Group/U-KAN)  

## Create conda virtual-environment
```bash
conda env create -f environment.yml
```

##  Build your own extreme precipitation dataset:
```
requirement:
1）Data from a 6-minute radar mosaic 
2）The size needs to be cropped to 256x256
1) complete and continuous data in the -1–3 h time period;
2) the presence of a mosaic sequence with echoes of magnitude 30 dBZ or more;
3) the echo reaches the hourly extreme precipitation in research area calculated;
4) sliding selection of samples at 30-minute intervals. 
```

## Train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=4 train_gpu.py
```
## Inference result：
```
python UKAN_output.py
```
