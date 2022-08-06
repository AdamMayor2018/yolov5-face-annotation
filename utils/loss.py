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
    # [1, 3] => [3, 1] => [3, num_targets] 三行 第一行num_target个0 第二行num_target个1 第三行num_target个2
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    # [num_targets, 16] [3, num_targets] -> [3, num_targets, 16] [3, num_targets, 1] -> [3, num_targets, 17]  17: [image_index+class+xywh+keypoints+anchor_index]
    # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
    # 先假设所有的target对三个anchor都是正样本(复制三份) 再进行筛选  并将ai加进去标记当前是哪个anchor的target
    # 通俗理解，现在就是每个yolo输出的feature map，都会放置target的坐标等信息。但因为有三个anchor，所以要把target放三层。
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
    # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
    # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
    g = 0.5  # bias  中心偏移  用来衡量target中心点离哪个格子更近
    # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lmgain
                        ], device=targets.device).float() * g  # offsets
    # 遍历三个feature 筛选每个feature map(包含batch张图片)的每个anchor的正样本
    for i in range(det.nl): # self.nl: number of detection layers   Detect的个数 = 3
        # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
        anchors = det.anchors[i] # [3, 2] 每一层对应放置三个anchor
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xywh gain
        #p:[32, 3, 100, 100, 16]、[32, 3, 50, 50, 16]、[32, 3, 25, 25, 16]  gain是[1, 1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] 之类的，为了把xy坐标放大到featuremap层面, class和image_index则不需要扩大
        gain[6:16] = torch.tensor(p[i].shape)[[3, 2, 3, 2, 3, 2, 3, 2, 3, 2]]  # 关键点 xyxy gain

        # 这里是将bbox和keypoint的坐标 都放缩到featuremap的尺度上 t:[3, num_targets, 17] 3个维度上的数值是完全一致的
        t = targets * gain #t:[3, num_targets, 17]
        if nt:
            # t: 3, num_targets, [image_index+class+xywh+keypoints+anchor_index]
            # Matches t:[3, num_targets, 2] / [3, 1, 2] => r:[3, num_targets, 2]
            r = t[:, :, 4:6] / anchors[:, None]  # wh相对于anchor的缩放 （w/w h/h）  r:[3, num_targets, 2]  [3, num_targets, 2] / [3, 1, 2] = [3, num_targets, 2]
            # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
            # torch.max(r, 1. / r)=[3, 63, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
            # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
            # j: [3, num_targets]  False: 当前gt是当前anchor的负样本  True: 当前gt是当前anchor的正样本   j标记了全部targets在每个featuremap上三个anchor视角下的正负样本的分布
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # hyp[anchor_t] 默认值是4.0， 也就是大到anchor宽高4倍，小到anchor宽高1/4倍的以内的都算正样本
            # 根据筛选条件j，过滤掉负样本。得到的是当前featuremap上三个anchor对应的全部正样本t（对应batch-size张图片）
            t = t[j]  # [3, num_targets, 17]  j:[3, num_targets] => [num_positive_targets, 17]  17里面是包含anchor_index的  相当于是把每个anchor对应的正样本的target拿出来了，然后又打散拼接到一起，理论上num_positive_targets也是<= 3*num_targets的
            # t:[num_positive_targets, image_index+class+xywh+keypoints+anchor_index] 打散没关系，是因为值里包含了anchor index
            # Offsets 筛选当前格子周围格子 找到2个离target中心最近的两个格子  可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
            # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 也就是说一个目标需要3个格子去预测(计算损失)
            # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个 用这三个格子去预测这个目标(计算损失)
            # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
            gxy = t[:, 2:4]  # grid xy t之前已经放缩到featuremap尺度了t: [num_positive_targets, 2]
            gxi = gain[[2, 3]] - gxy  # 转换为target中心点相对于右下角的坐标
            # 筛选中心坐标 距离当前grid_cell的左、上方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
            # j: [num_positive_targets] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
            # k: [num_positive_targets] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            # 筛选中心坐标 距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
            # l: [num_positive_targets] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
            # m: [num_positive_targets] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            # j: [5, num_positive_targets]  torch.ones_like(j): 当前格子, 不需要筛选全是True  j, k, l, m: 左上右下格子的筛选结果
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            # t<=3*num_positive_targets 因为是target中心格点和最近的两个格点 当且仅当target不在边上的时候成立
            t = t.repeat((5, 1, 1))[j]  #j:[5, num_positive_targets] t:[num_positive_targets, 17] => [5, num_positive_targets, 17] => [每个anchor格点及最多两个最近格点 * num_positive_targets, 17] 后面记为num_predict_target
            # 经过j筛选后t：[7269, 17] 这里相当于把没有出界的target(正样本)的
            #(torch.zeros_like(gxy)[None] + off[:, None]):[5, num_positive_targets, 2]  / j:[5, num_positive_targets] => offsets:[后面记为num_predict_target, 2]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]    #gxy: [num_positive_targets, 2] off:[5, 2] => [5, num_positive_targets, 2]    offsets:[]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class [num_predict_target]
        gxy = t[:, 2:4]  # grid xy [num_predict_target, 2]
        gwh = t[:, 4:6]  # grid wh [num_predict_target, 2]
        gij = (gxy - offsets).long()  # 预测真实框的网格所在的左上角坐标(有左上右下的网格)
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 16].long()  # anchor indices
        # # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        # xywh 其中xy为这个target对当前grid_cell左上角的偏移量
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
