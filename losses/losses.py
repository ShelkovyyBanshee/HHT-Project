import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, clf_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.clf_weight = clf_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, seg_pred, seg_target, clf_pred, clf_target):
        bce_loss = self.bce(seg_pred, seg_target)
        d_loss = dice_loss(seg_pred, seg_target)

        seg_loss = self.bce_weight * bce_loss + self.dice_weight * d_loss
        clf_loss = self.ce(clf_pred, clf_target)

        total = seg_loss + self.clf_weight * clf_loss

        return total, seg_loss, clf_loss
