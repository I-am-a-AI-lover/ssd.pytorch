# -*- coding: utf-8 -*-
import torch
"""
prior_box 的表示为(x_center,y_center,w,h)
gruth_box 的表示为(xmin, ymin, xmax, ymax)
"""

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.

    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
        center_x,center_y,w,h
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):# 在程序中没用到
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):# 交并比
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
        threshold: (float)iou阈值
        truths: (tensor) 真实标签的box, [num_obj, 4].
        priors: (tensor) 每层的defaut Prior_boxes , [num_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,[num_priors, 4].
        labels: (tensor) 从本地文件中得到的label,故暂时没有＋１: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) 每个defaut Prior_boxes匹配真实标签的类别[num_priors]
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    """
    1.是先将真实标签框分配给iou最大的default box，确保每个标签至少有一个default box可以匹配
    但不能给一个框匹配两个标签
    2.对于剩下的未匹配的先验框，若与某个真实标签框的 IOU 大于某个阈值（一般取0.5），则该先验框也与 ground truth 匹配
    如果一个默认框和两个真实标签框的iou都大于0.5，则选择iou最大的那个
    ps：选择先碰到的也可以
    """

    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    ) # [num_obj,num_priors]
    # (Bipartite Matching)

    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    #best_prior_overlap[num_obj,1]best prior for each ground truth
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    # best_truth_overlap[1,num_priors]best ground truth for each prior
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # 将２添加到best_truth_overlap的０维度上索引为best_prior_idx的位置上
    # 给个真实标签都有一个最大ＩＯＵ的prior_box将该prior_box设置为２,即一个最大值
    # 与conf[best_truth_overlap < threshold] = 0对应

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    #　确保每个prior_box只能匹配一个真实标签
    for j in range(best_prior_idx.size(0)):#这里一般不会起作用，除非一个prior_box同时与多个标签iou最大
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    """
    matches = truths[best_truth_idx]
    truth [snum_obj, 4], best_truth_idx [num_priors]
    matches [num_priors,4]
    用best_truth_idx的值重复truth的行，
    如best_truth_idx为[0,1,0]
    则matches的第一行为truth的第一行，第二行为truth的第二行，第三行为truth的第一行，
    则第i行是离第ｊ个priors_box最近的truth_box的坐标，这样方便encode(matches, priors, variances)
    """
    conf = labels[best_truth_idx] + 1 # Shape: [num_priors],与上一行一样


    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):

    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    在训练阶段将ground truth boxes转化为与default_prior boxes之间的off set
    因为location regression返回的是off set

    重要一点：　ground truth boxes为[xmin, ymin, xmax, ymax],
            　default_prior boxes为[x_center,y_center,w,h]

            　
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
#%%
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

#%%
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    解码从locations  predictions层获得的boxes
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        如果有多分类需要将ｎｍｓ放在循环里面
        先选取置信度前top_k个框再进行nms
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    #　一个一维矩阵，个数为scores的张量个数
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    #[x1,y1,x2,y2] = [x_min,y_min,x_max,y_max]

    area = torch.mul(x2 - x1, y2 - y1) #对应位相乘
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    # 从小到大排序，取倒数top_k个
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val目前最大值的索引
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        """
        沿着指定维度对输入进行切片。
            参数：
                input (Tensor) – 输入张量
                dim (int) – 索引的轴
                index (LongTensor) – 包含索引下标的一维张量
                out (Tensor, optional) – 目标张量
        """
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score

        """
        clamp将输入input张量每个元素夹紧到区间 [min,max]，并返回结果到一个新张量。
        """
        # 计算当前最大置信框与其他剩余框的交集，作者这段代码写的不好，容易误导
        xx1 = torch.clamp(xx1, min=x1[i])  # max(x1[i],xx1)
        yy1 = torch.clamp(yy1, min=y1[i])  # max(y1[i],yy1)
        xx2 = torch.clamp(xx2, max=x2[i])  # min(x2[i],xx2)
        yy2 = torch.clamp(yy2, max=y2[i])  # min(y2[i],yy2)
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
        h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
        w = torch.clamp(w, min=0.0)  # max(w,0)
        h = torch.clamp(h, min=0.0)  # max(h,0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
        """
        torch.le(input, other, out=None) → Tensor 逐元素比较input和other ， 即是否input<=other
        返回1或0的tensor结果
        实际上比较的是IoU与overlap
        """
    return keep, count
