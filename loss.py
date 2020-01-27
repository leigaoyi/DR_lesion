import torch.nn.functional as F
import torch
import torch.nn as nn



# for metric code: https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py

#cross_entropy = F.cross_entropy
device = torch.device('cuda:0')
#loss_weight = torch.tensor([1., 2000.]).to(device)
cross_entropy = nn.CrossEntropyLoss()

def weighted_dice(output, target, type_weight='square'):
    # output [BS, 2, w, h], target [BS, w, h]
#    prediction = F.softmax(output, dim=1)[:, 1, :, :]
#    prediction = prediction.float()
            
    prediction = output
    assert target.max().item() < 1.1
    
    ref_vol = target.sum(0)
    intersect = (target*prediction).sum(0)
    seg_vol = prediction.sum(0)
    
    if type_weight == 'square':
        weights = ref_vol ** 2
        weights = weights.float()
        torch.reciprocal(weights)
    else:
        weights = torch.ones_like(ref_vol)
    
    new_weights = torch.where(torch.isinf(weights), torch.zeros_like(weights), weights)
    #print(new_weights)
    #print(new_weights.max(0))
    weights = torch.where(torch.isinf(weights), torch.ones_like(weights)*(new_weights.max()), weights)
    
    numerator = 2 * (weights*intersect).sum()
    denomenator = (weights*(ref_vol+seg_vol)).sum() + 1e-6
    
    dice = numerator/denomenator
    
    return 1 - dice
        
    
def multi_dice(pred, target, class_num = 2):
    pred = F.softmax(pred, dim=1).float()
    pred_list = [pred[:, 0, :, :], pred[:, 1, :, :]]
    target_list = [1-target, target]
    
    dice = 0
    for i in range(class_num):
        dice += weighted_dice(pred_list[i], target_list[i], type_weight='square')
    return dice/class_num

def multi_cc_dice(pred, target, class_num = 2):
    pred = F.softmax(pred, dim=1).float()
    pred_list = [pred[:, 0, :, :], pred[:, 1, :, :]]
    target_list = [1-target, target]
    
    dice = 0
    for i in range(class_num):
        dice += weighted_dice(pred_list[i], target_list[i], type_weight='same')
    return dice/class_num
    
def cc_cross_entropy(output, target, class_num=2):
    pred = F.softmax(output, dim=1).float()
    
    return F.cross_entropy(pred, target)    


def hard_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    Np, Nn = pos.numel(), neg.numel()

    pos = pos.sum()

    k = min(Np*alpha, Nn)
    if k > 0:
        neg, _ = torch.topk(neg, int(k))
        neg = neg.sum()
    else:
        neg = 0.0

    loss = (pos + neg)/ (Np + k)

    return loss


def hard_per_im_cross_entropy(output, target, alpha=3.0):
    n, c = output.shape[:2]
    output = output.view(n, c, -1)
    target = target.view(n, -1)

    mtx = F.cross_entropy(output, target, reduce=False)

    pos = target > 0
    num_pos = pos.long().sum(dim=1, keepdim=True)

    loss = mtx.clone().detach()
    loss[pos] = 0
    _, loss_idx = loss.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    num_neg = torch.clamp(alpha*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg

    return mtx[neg + pos].mean()


def focal_loss(output, target, alpha=0.25, gamma=2.0):
    n = target.size(0)

    lsfm = F.cross_entropy(output, target, reduce=False)

    pos = (target > 0).float()
    Np  = pos.view(n, -1).sum(1).view(n, 1, 1, 1)

    Np  = torch.clamp(Np, 1.0)
    z   = pos * alpha / Np / n  + (1.0 - pos) * (1.0 - alpha) / Np / n
    z   = z.detach()

    focal = torch.pow(1.0 - torch.exp(-lsfm), gamma) * lsfm * z

    return focal.sum()


def mean_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    pos = pos.mean() if pos.numel() > 0 else 0
    neg = neg.mean() if pos.neg() > 0 else 0

    loss = (neg * alpha + pos)/(alpha + 1.0)
    return loss




eps = 0.1
def dice(output, target):
    num = 2*(output*target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def cross_entropy_dice(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 5):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice(o, t)

    return loss

# in original paper: class 3 is ignored
# https://github.com/MIC-DKFZ/BraTS2017/blob/master/dataset.py#L283
# dice score per image per positive class, then aveg
def dice_per_im(output, target):
    n = output.shape[0]
    output = output.view(n, -1)
    target = target.view(n, -1)
    num = 2*(output*target).sum(1) + eps
    den = output.sum(1) + target.sum(1) + eps
    return 1.0 - (num/den).mean()

def cross_entropy_dice_per_im(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 5):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice_per_im(o, t)

    return loss


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss
