# Copyright (c) OpenMMLab. All rights reserved.

import os
import tomli

#import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import multi_gpu_test

from app_class import App

def main():
    
    with open("args.toml", mode = "rb") as argsF:
        args = tomli.load(argsF)
        
    model_filename = args["model_filename"]
    print(f"Loading Model from {model_filename}...\n")
    model = torch.load(open(model_filename, 'rb'))
    
    data_loader_filename = args["dataloader_filename"]
    print(f"Loading DataLoader from {data_loader_filename}...\n")
    data_loader = torch.load(open(data_loader_filename, 'rb'))
    
    if args["launcher"] == 'none':
        distributed = False
    else:
        distributed = True
        
    gpu_ids = [args["gpu_id"]]
    
    if not distributed:
        model = MMDataParallel(model, device_ids = gpu_ids)
    else:
        model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],broadcast_buffers=False)
        
    model.eval()
    app = App(model, data_loader)
    app.mainloop()


    


if __name__ == '__main__':
    main()



