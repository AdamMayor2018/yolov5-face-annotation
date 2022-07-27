# Loss functions

import torch
import torch.nn as nn
import numpy as np
from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf   Figure 5
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t==-1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()

class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()#nn.SmoothL1Loss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, truel, mask):
        loss = self.loss_fcn(pred*mask, truel*mask)
        return loss / (torch.sum(mask) + 10e-14)


def compute_loss(p, targets, model):  # predictions, targets, model
    """
        train.py里负责loss计算
    :param p: 预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
           tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
           如: [32, 3, 100, 100, 16]、[32, 3, 50, 50, 16]、[32, 3, 25, 25, 16]
           [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
           可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
    :param targets: 数据增强后的真实框【num_object, batch_index+class+xywh+keypoints】
    :param model:
    :return:
    """
    device = targets.device
    # lcls：分类损失 lbox： bbox回归损失 lobj：有无目标损失 lmark：关键点回归损失
    lcls, lbox, lobj, lmark = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors, tlandmarks, lmks_mask = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))  # weight=model.class_weights)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

    landmarks_loss = LandmarksLoss(1.0)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    no = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

            # Classification
            if model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 15:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 15:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            #landmarks loss
            #plandmarks = ps[:,5:15].sigmoid() * 8. - 4.
            plandmarks = ps[:,5:15]

            plandmarks[:, 0:2] = plandmarks[:, 0:2] * anchors[i]
            plandmarks[:, 2:4] = plandmarks[:, 2:4] * anchors[i]
            plandmarks[:, 4:6] = plandmarks[:, 4:6] * anchors[i]
            plandmarks[:, 6:8] = plandmarks[:, 6:8] * anchors[i]
            plandmarks[:, 8:10] = plandmarks[:,8:10] * anchors[i]

            lmark += landmarks_loss(plandmarks, tlandmarks[i], lmks_mask[i])


        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / no  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if no == 4 else 1.)
    lcls *= h['cls'] * s
    lmark *= h['landmark'] * s

    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls + lmark
    return loss * bs, torch.cat((lbox, lobj, lcls, lmark, loss)).detach()


def build_targets(p, targets, model):
    """
        输出gt的信息
    :param p:      预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                   如: [32, 3, 100, 100, 16]、[32, 3, 50, 50, 16]、[32, 3, 25, 25, 16]
                   [bs, anchor_num, grid_h, grid_w, class(1)+xywh(4)+keypoints(10)]
    :param targets:[num_target, img_index+class_index+xywh(normalized)+keypoints(normalized)] 比如[957, 16]
    :param model:
    :return:
    """
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch, landmarks, lmks_mask = [], [], [], [], [], []
    #gain = torch.ones(17, device=targets.device)  # normalized to gridspace gain
    # 17: image_index + class + xywh + keypoints
    # gain是为了后面将targets=[na,nt,17]中的归一化了的xywh映射到相对feature map尺度上
    gain = torch.ones(17, device=targets.device)
    # ai 【3, num_targets】 需要在3个anchor上都进行训练 所以将标签赋值na=3个 ai代表3个anchor上在所有target对应的anchor索引。
    # 用来标记当前这个target属于哪个anchor
    # [1, 3] => [3, 1] => [3, num_targets] 三行 第一行63个0 第二行62个1 第三行63个2
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # [63, 6] [3, 63] -> [3, 63, 6] [3, 63, 1] -> [3, 63, 7]  7: [image_index+class+xywh+anchor_index]
    # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
    # 先假设所有的target对三个anchor都是正样本(复制三份) 再进行筛选  并将ai加进去标记当前是哪个anchor的target
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        #landmarks 10
        gain[6:16] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 16].long()  # anchor indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

        #landmarks
        lks = t[:,6:16]
        #lks_mask = lks > 0
        #lks_mask = lks_mask.float()
        lks_mask = torch.where(lks < 0, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))

        #应该是关键点的坐标除以anch的宽高才对，便于模型学习。使用gwh会导致不同关键点的编码不同，没有统一的参考标准

        lks[:, [0, 1]] = (lks[:, [0, 1]] - gij)
        lks[:, [2, 3]] = (lks[:, [2, 3]] - gij)
        lks[:, [4, 5]] = (lks[:, [4, 5]] - gij)
        lks[:, [6, 7]] = (lks[:, [6, 7]] - gij)
        lks[:, [8, 9]] = (lks[:, [8, 9]] - gij)

        '''
        #anch_w = torch.ones(5, device=targets.device).fill_(anchors[0][0])
        #anch_wh = torch.ones(5, device=targets.device)
        anch_f_0 = (a == 0).unsqueeze(1).repeat(1, 5)
        anch_f_1 = (a == 1).unsqueeze(1).repeat(1, 5)
        anch_f_2 = (a == 2).unsqueeze(1).repeat(1, 5)
        lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_0, lks[:, [0, 2, 4, 6, 8]] / anchors[0][0], lks[:, [0, 2, 4, 6, 8]])
        lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_1, lks[:, [0, 2, 4, 6, 8]] / anchors[1][0], lks[:, [0, 2, 4, 6, 8]])
        lks[:, [0, 2, 4, 6, 8]] = torch.where(anch_f_2, lks[:, [0, 2, 4, 6, 8]] / anchors[2][0], lks[:, [0, 2, 4, 6, 8]])

        lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_0, lks[:, [1, 3, 5, 7, 9]] / anchors[0][1], lks[:, [1, 3, 5, 7, 9]])
        lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_1, lks[:, [1, 3, 5, 7, 9]] / anchors[1][1], lks[:, [1, 3, 5, 7, 9]])
        lks[:, [1, 3, 5, 7, 9]] = torch.where(anch_f_2, lks[:, [1, 3, 5, 7, 9]] / anchors[2][1], lks[:, [1, 3, 5, 7, 9]])

        #new_lks = lks[lks_mask>0]
        #print('new_lks:   min --- ', torch.min(new_lks), '  max --- ', torch.max(new_lks))
        
        lks_mask_1 = torch.where(lks < -3, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))
        lks_mask_2 = torch.where(lks > 3, torch.full_like(lks, 0.), torch.full_like(lks, 1.0))

        lks_mask_new = lks_mask * lks_mask_1 * lks_mask_2
        lks_mask_new[:, 0] = lks_mask_new[:, 0] * lks_mask_new[:, 1]
        lks_mask_new[:, 1] = lks_mask_new[:, 0] * lks_mask_new[:, 1]
        lks_mask_new[:, 2] = lks_mask_new[:, 2] * lks_mask_new[:, 3]
        lks_mask_new[:, 3] = lks_mask_new[:, 2] * lks_mask_new[:, 3]
        lks_mask_new[:, 4] = lks_mask_new[:, 4] * lks_mask_new[:, 5]
        lks_mask_new[:, 5] = lks_mask_new[:, 4] * lks_mask_new[:, 5]
        lks_mask_new[:, 6] = lks_mask_new[:, 6] * lks_mask_new[:, 7]
        lks_mask_new[:, 7] = lks_mask_new[:, 6] * lks_mask_new[:, 7]
        lks_mask_new[:, 8] = lks_mask_new[:, 8] * lks_mask_new[:, 9]
        lks_mask_new[:, 9] = lks_mask_new[:, 8] * lks_mask_new[:, 9]
        '''
        lks_mask_new = lks_mask
        lmks_mask.append(lks_mask_new)
        landmarks.append(lks)
        #print('lks: ',  lks.size())

    return tcls, tbox, indices, anch, landmarks, lmks_mask
