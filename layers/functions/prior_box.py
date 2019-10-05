from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """
    Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    """ 
    cfg = 
        voc = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [38, 19, 10, 5, 3, 1], 
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'VOC',
        }
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps'] # steps为映射比例,不是步长
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # itertools.product(a[,repeat=1])是a的笛卡尔乘机
                # 此处,for i, j in product(range(f), repeat=2):
                # 等于for i in range(f):
                #       for j in range(f):

                """
                以feature map上每个点(i,j)的中点为中心（i+0.5,j+0.5）
                然后中心点的坐标会乘以step，相当于从feature map位置映射回原图位置
                再缩放成0-1的相对距离
                原始公式应该为cx = (j+0.5) * step /min_dim，这里拆分成两步计算
                """
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #每个box都有两个默认的正方形尺寸
                # aspect_ratio: 1 正方形1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1 正方形2
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                # 针对每层设置的ratio参数，产生其他2个或者4个ratio为[1/2,2/1,1/3,/3/1]的默认框
                # 其中短边为s_k/sqrt(ar),长边为 s_k*sqrt(ar),其中s_k为该层min_sizes/image_size

                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
            # 输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
            # 大于1时取1,小于0时取0,其余不变
        return output
