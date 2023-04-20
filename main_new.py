# Copyright (c) OpenMMLab. All rights reserved.

import os
import tomli

#import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import multi_gpu_test

from App_new import App

def main():
    
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()



