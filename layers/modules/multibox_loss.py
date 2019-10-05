# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap  #需要训练的负正样本比例
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        输入:
            predictions (tuple): 一个三元素的元组，包含了预测信息.
                loc_data [batch,num_priors,4] 所有预测框的offsets.
                conf_data [batch,num_priors,num_classes] 所有预测框的分类置信度.
                priors [num_priors,4] 所有默认框的位置

            targets [batch,num_objs,5] 所有真实目标的信息 5: [xmin, ymin, xmax, ymax, label_idx]

        返回：
            loss_l, loss_c：定位损失和分类损失
        """

        loc_data, conf_data, priors = predictions
        num = loc_data.size(0) #batch_size
        priors = priors[:loc_data.size(1), :] #多余的　 priors == priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        loc_t = torch.Tensor(num, num_priors, 4)
        # [batch, num_priors, 4] 匹配到的真实目标和默认框之间的offset，是learning target
        conf_t = torch.LongTensor(num, num_priors)
        # [batch, num_priors] 匹配后默认框的类别，是learning target

        # 对于batch中的每一个图片进行匹配
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        #正样本查找，等于0为背景 [batch, num_priors]
        pos = conf_t > 0 #postive #返回[batch, num_priors]的tensorbool值
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        """
        pos.dim() 返回pos的维数，因为pos为[batch, num_priors]，故pos.dim()＝２
        pos.unsqueeze(pos.dim())在２维上再加一维，为[batch, num_priors,１]
        expand_as(loc_data)将数据扩展为与loc_data一样的数据大小[batch, num_priors，4]
        由于４为box的坐标，故第三维的数字可以和原始第二维的数字一.即原始[i,j] = m,则[i,j,k] = m,其中k={0,1,2,3}
        这样操作为了下面的索引
        """

        """
        Localization Loss (Smooth L1)
          loc_t代表一对匹配的真实框和默认框的offset
          loc_p代表预测框和默认框之间的offset
          
        """
        loc_p = loc_data[pos_idx].view(-1, 4) #[num_pos_priors,4],num_pos_priors为所有匹配到正类标签的priors_box
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) #　回归损失，只考虑正类的

        # Compute max conf across batch for hard negative mining
        # 难负样本挖掘的依据：loss_c损失较大
        batch_conf = conf_data.view(-1, self.num_classes) # [batch*num_priors , num_classes]

        # [batch，num_priors] 计算所有默认框的分类损失
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0    # [batch*num_priors] 因为是给负样本排序的，所以手动给正样本损失置0
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)#[batch,1]
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # self.negpos_ratio = 3
        neg = idx_rank < num_neg.expand_as(idx_rank) #返回负类briors_box的tensorbool值

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)#[]
        targets_weighted = conf_t[(pos+neg).gt(0)]#[]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
