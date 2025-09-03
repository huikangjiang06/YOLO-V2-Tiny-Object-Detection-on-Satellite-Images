import os
os.chdir("/root/Day4/object_detection")

import torch

from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip
from utils.plot_utils import plot_loss_and_lr, plot_map
from utils.train_utils import train_one_epoch, write_tb, create_model

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import pickle
import os
import sys
import itertools
import json
import os
import numpy as np
import torch
from PIL import Image
import torch
import math
import pdb
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Config:

    # data transform parameter
    train_horizon_flip_prob = 0.0  # data horizon flip probility in train transform
    min_size = 800
    max_size = 1000
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # anchor parameters
    anchors = [(10,20),(10,10),(20,10),(32,64),(64,32),(64,64)]
    num_anchors = len(anchors)

    # data parameters
    num_classes = 4
    image_size = 640

    # loss parameters
    noobject_scale = 0.01
    object_scale = 0.01
    coord_scale = 1
    class_scale = 1

    # iou parameters
    thresh = 0.45

    # training parameters
    device_name = 'cuda'
    start_epoch = 0  # start epoch
    num_epochs = 30  # train epochs
    lr = 1e-3
    momentum = 0.9
    weight_decay = 0.0005

    # learning rate schedule
    lr_gamma = 0.33
    lr_dec_step_size = 100
    batch_size = 16
    
    # dataset
    train_anno_path = "data/SkyFusion/train/_annotations.coco.json"
    train_image_dir = "data/SkyFusion/train/"
    valid_anno_path = "data/SkyFusion/valid/_annotations.coco.json"
    valid_image_dir = "data/SkyFusion/valid/"
    test_anno_path = "data/SkyFusion/test/_annotations.coco.json"
    test_image_dir = "data/SkyFusion/test/"

cfg = Config()

class coco(torch.utils.data.Dataset):
    def __init__(self, anno_path, image_dir, transforms=None):

        self._anno_path = anno_path
        self._image_dir = image_dir
        self._transforms = transforms

        with open(self._anno_path) as anno_file:
            self.anno = json.load(anno_file)
            
        cats = self.anno['categories']
        
        cats = sorted(cats, key=lambda x: x['id'])
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        
        self.classes = self._classes
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats], [c['id'] for c in cats])))
        self.coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls], self._class_to_ind[cls]) for cls in self._classes[1:]])
        
        # print(cats)
        # print(self._class_to_ind)
        # print(self._class_to_coco_cat_id)
        # print(self.coco_cat_id_to_class_ind)

    def __len__(self):
        return len(self.anno['images'])

    def __getitem__(self, idx):
        a = self.anno['images'][idx]
        image_idx = a['id']
        image_file_name = a['file_name']
        img_path = os.path.join(self._image_dir, image_file_name)
        image = Image.open(img_path)

        width = a['width']
        height = a['height']

        iscrowd = None  # 与原始代码一致
        annIds = []
        objs = []
        
        # 遍历所有标注，筛选出属于当前图片的标注
        for ann in self.anno['annotations']:
            # 检查图片ID是否匹配
            if ann['image_id'] == image_idx:
                # 检查 iscrowd 条件（None 表示不限制）
                if iscrowd is None or ann['iscrowd'] == iscrowd:
                    annIds.append(ann['id'])
                    objs.append(ann)

        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.float32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)

        iscrowd = []
        for ix, obj in enumerate(objs):
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            iscrowd.append(int(obj["iscrowd"]))

        # convert everything into a torch.Tensor
        image_id = torch.tensor([image_idx])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        gt_classes = torch.as_tensor(gt_classes, dtype=torch.int32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes.float(), "labels": gt_classes.float(), "image_id": image_id.long(), "area": area.float(), "iscrowd": iscrowd.float(), "num_objs": torch.tensor([num_objs],dtype=torch.int32)}

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        padding_rows = 150 - target["boxes"].size(0)
        if padding_rows > 0:
            zeros = torch.zeros(padding_rows,4).float()
            target["boxes"] = torch.cat([target["boxes"],zeros],dim=0).to(torch.float32)
            zeros = torch.zeros(padding_rows).float()
            for k in ["labels","area","iscrowd"]:
                target[k] = torch.cat([target[k],zeros],dim=0).to(torch.float32)
        elif padding_rows < 0:
            for k in ["boxes","labels","area","iscrowd"]:
                target[k] = target[k][:150].to(torch.float32)
            target["num_objs"] = torch.tensor([150],dtype=torch.int32)
        
        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @property
    def class_to_coco_cat_id(self):
        return self._class_to_coco_cat_id


device = torch.device(cfg.device_name)
print("Using {} device training.".format(device.type))

data_transform = {
    "train": Compose([ToTensor(), RandomHorizontalFlip(cfg.train_horizon_flip_prob)]),
    "val": Compose([ToTensor()])
}

# load train data set
train_data_set = coco(cfg.train_anno_path, cfg.train_image_dir, data_transform["train"])
batch_size = cfg.batch_size
print(batch_size)
train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=8)

# load validation data set
val_data_set = coco(cfg.valid_anno_path, cfg.valid_image_dir, data_transform["val"])
val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=8)


### passthrough部分的处理
def reorg(x, stride_h=2, stride_w=2):
    batch_size, channels, height, width = x.size()
    _height, _width = height // stride_h, width // stride_w
    if 1:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, stride_h * stride_w).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, stride_h * stride_w, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)
    else:
        x = x.view(batch_size, channels, _height, stride_h, _width, stride_w)
        x = x.permute(0, 1, 3, 5, 2, 4) # batch_size, channels, stride, stride, _height, _width
        x = x.contiguous()
        x = x.view(batch_size, -1, _height, _width)
    return x


### 卷积块
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding='same')
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

### 模型
class YoloV2(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 anchors=cfg.anchors,
                 num_classes=cfg.num_classes,
                 filter_thres = 0.5,
                 nms_thres = 0.45,
                ):
        nn.Module.__init__(self)
        """
        model output：H * W * ( num_anchors * ( 5 + num_classes ) )
        """

        self.num_classes = num_classes
        self.anchors = anchors
        self.filter_thres = filter_thres
        self.nms_thres = nms_thres
        self.out_channels = len(anchors) * (5 + num_classes)
        
        layers = []

        # two 3*3 convolution layers
        layers.append(Conv2d(in_channels, 32, kernel_size = 3))
        layers.append(nn.MaxPool2d(kernel_size = 2))
        
        layers.append(Conv2d(32, 64, kernel_size = 3))
        layers.append(nn.MaxPool2d(kernel_size = 2))

        # two 3*3+1*1+3*3 convolution blocks
        layers.append(Conv2d(64, 128, kernel_size = 3))
        layers.append(Conv2d(128, 64, kernel_size = 1))
        layers.append(Conv2d(64, 128, kernel_size = 3))
        layers.append(nn.MaxPool2d(kernel_size=2))
        
        layers.append(Conv2d(128, 256, kernel_size = 3))
        layers.append(Conv2d(256, 128, kernel_size = 1))
        layers.append(Conv2d(128, 256, kernel_size = 3))
        self.layers1 = nn.Sequential(*layers) # 实例化

        # 在此处把输入拿走进行passthrough

        layers = []

        # 升到512维度之后，先pssthrough，再MaxPool
        layers.append(nn.MaxPool2d(kernel_size=2))

        # some large 3*3 and 1*1 combo
        layers.append(Conv2d(256, 512, kernel_size = 3))
        layers.append(Conv2d(512, 256, kernel_size = 1))
        layers.append(Conv2d(256, 512, kernel_size = 3))
        layers.append(Conv2d(512, 256, kernel_size = 1))
        layers.append(Conv2d(256, 512, kernel_size = 3))
        self.layers2 = nn.Sequential(*layers)

        # 在这里吧passthrough的结果拿来concat

        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2))

        # 更多的 3*3 + 1*1
        layers.append(Conv2d(1536, 1024, kernel_size = 3))
        layers.append(Conv2d(1024, 512, kernel_size = 1))
        layers.append(Conv2d(512, 1024, kernel_size = 3))
        layers.append(Conv2d(1024, 512, kernel_size = 1))
        layers.append(Conv2d(512, 1024, kernel_size = 3))
        layers.append(Conv2d(1024, self.out_channels, kernel_size = 1))
        self.layers3 = nn.Sequential(*layers)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layers1(x)
        _x = reorg(x)
        x = self.layers2(x)
        x = torch.cat([_x, x], 1)
        x = self.layers3(x)
        return x


def box_ious(box1, box2): ### input B * (x1, y1, x2, y2)
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """
    with torch.autograd.set_detect_anomaly(True):

        N = box1.size(0)
        K = box2.size(0)
    
        # 创建显式副本
        box1_x1 = box1[:, 0].clone().reshape(N, 1)
        box2_x1 = box2[:, 0].clone().reshape(1, K)
        xi1 = torch.max(box1_x1, box2_x1)

        box1_y1 = box1[:, 1].clone().reshape(N, 1)
        box2_y1 = box2[:, 1].clone().reshape(1, K)
        yi1 = torch.max(box1_y1, box2_y1)

        box1_x2 = box1[:, 2].clone().reshape(N, 1)
        box2_x2 = box2[:, 2].clone().reshape(1, K)
        xi2 = torch.max(box1_x2, box2_x2)

        box1_y2 = box1[:, 3].clone().reshape(N, 1)
        box2_y2 = box2[:, 3].clone().reshape(1, K)
        yi2 = torch.max(box1_y2, box2_y2)
        
        iw = torch.max(xi2 - xi1, torch.tensor(0.0, device=box1.device))
        ih = torch.max(yi2 - yi1, torch.tensor(0.0, device=box1.device))
    
        inter = iw * ih
    
        # 计算面积时使用非 inplace 操作
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
        # 使用 view 而不是 reshape
        box1_area = box1_area.view(N, 1)
        box2_area = box2_area.view(1, K)
    
        union_area = box1_area + box2_area - inter
    
        # 添加小量防止除以0（非 inplace 操作）
        ious = inter / (union_area + 1e-10)
    
        return ious

def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box


def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors

def build_target(output, gt_data):
    """
    Build the training target for output tensor

    Arguments:

    output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
    gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

    delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

    gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values(x1, y1, x2, y2) range 0~1
    gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
    num_obj_batch -- tensor of shape (B, 1). number of objects


    Returns:
    iou_target -- tensor of shape (B, H * W * num_anchors, 1)
    iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
    box_target -- tensor of shape (B, H * W * num_anchors, 4) 
    box_mask -- tensor of shape (B, H * W * num_anchors, 1)
    class_target -- tensor of shape (B, H * W * num_anchors, 1)
    class_mask -- tensor of shape (B, H * W * num_anchors, 1)

    """
    output = output.cuda()
    bsize = output.size(0)
    H = output.size(2)
    W = output.size(3)

    output = output.permute(0,2,3,1).reshape(-1,H,W,cfg.num_anchors,5+cfg.num_classes).reshape(-1,H*W*cfg.num_anchors,5+cfg.num_classes)
    
    xy_pred = torch.sigmoid(output[:,:,1:3])
    hw_pred = torch.exp(output[:,:,3:5])
    delta_pred_batch = torch.cat([xy_pred,hw_pred],dim=-1)
    
    conf_pred_batch = torch.sigmoid(output[:,:,0])
    #class_score_batch = torch.sigmoid(output[:,:,5:5+cfg.num_classes])

    gt_boxes_batch = gt_data[0].cuda() / cfg.image_size
    gt_classes_batch = gt_data[1].cuda()
    num_boxes_batch = gt_data[2].cuda()

    ### output base
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, cfg.num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, cfg.num_anchors, 1)) * cfg.noobject_scale

    box_target = delta_pred_batch.new_zeros((bsize, H * W, cfg.num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, cfg.num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, cfg.num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, cfg.num_anchors, 1))

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    anchors = torch.FloatTensor(cfg.anchors)/32
    all_grid_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    ### 遍历batch中所有样本
    for b in range(bsize):
        delta_pred = delta_pred_batch[b]
        num_obj = num_boxes_batch[b].item()
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= W 
        gt_boxes[:, 1::2] *= H

        # step 1: process IoU target

        # apply delta_pred to pre-defined anchors
        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        box_pred = box_transform_inv(all_grid_xywh, delta_pred) ### 生成最终的预测边框
        box_pred = xywh2xxyy(box_pred)

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        ious = box_ious(box_pred, gt_boxes) # shape: (H * W * num_anchors, num_obj)
        ious = ious.view(-1, cfg.num_anchors, num_obj)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True) # shape: (H * W, num_anchors, 1)

        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            iou_mask[b][max_iou >= cfg.thresh] = 0

        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, cfg.num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects

        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU

            gt_box_xywh = gt_boxes_xywh[t]
            gt_class = gt_classes[t]
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            cell_idx = cell_idx_y * W + cell_idx_x
            cell_idx = cell_idx.long()
            
            # update box_target, box_mask
            overlaps_in_cell = overlaps[cell_idx, :, t]
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            assigned_grid = all_grid_xywh.view(-1, cfg.num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box)
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    return iou_target.view(bsize, -1, 1), \
           iou_mask.view(bsize, -1, 1), \
           box_target.view(bsize, -1, 4),\
           box_mask.view(bsize, -1, 1), \
           class_target.view(bsize, -1, 1).long(), \
           class_mask.view(bsize, -1, 1)


def yolo_loss(output, target):
    """
    Build yolo loss

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """
    #bsize = output.size(0)
    H = output.size(2)
    W = output.size(3)

    output = output.permute(0,2,3,1).reshape(-1,H,W,cfg.num_anchors,5+cfg.num_classes).reshape(-1,H*W*cfg.num_anchors,5+cfg.num_classes)
    
    xy_pred = torch.sigmoid(output[:,:,1:3])
    hw_pred = torch.exp(output[:,:,3:5])
    delta_pred_batch = torch.cat([xy_pred,hw_pred],dim=-1)
    
    conf_pred_batch = torch.sigmoid(output[:,:,0]).unsqueeze(-1)
    class_score_batch = torch.sigmoid(output[:,:,5:5+cfg.num_classes])

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]
    
    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask, reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
 
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
 
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def Self_NMS(boxes, scores, iou_thres, GIoU=False, DIoU=False, CIoU=False):
    """
    :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
    :param scores: (Tensor[N]): scores for each one of the boxes
    :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
    :return:keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    # 按conf从大到小排序
    B = torch.argsort(scores, dim=-1, descending=True) ##返回的是索引值
    keep = []
    while B.numel() > 0:
        # 取出置信度最高的
        index = B[0]
        keep.append(index)
        if B.numel() == 1: break
        # 计算iou,根据需求可选择GIOU,DIOU,CIOU
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
        # 找到符合阈值的下标
        inds = torch.nonzero(iou <= iou_thres).reshape(-1)
        ##这里主要是处理一个索引的问题，这里计算iou的时候是用score最高的box与其他box计算，得到的iou是个列表
        ##但此时的iou列表已经比B少了一个值了，ins返回的是iou < 阈值的框在iou列表里面的索引
        ##这时要返回其在B中的索引就需要把inds+1，然后得到进行一轮NMS后剩下的框
        B = B[inds + 1]
    return torch.tensor(keep)

def Full_NMS(output):
    B = output.size(0)
    H = output.size(-1)
    W = output.size(-2)
    grid_size = cfg.image_size / H
    output = output.permute(0,2,3,1).reshape(-1,H,W,cfg.num_anchors,5+cfg.num_classes).reshape(-1,H*W*cfg.num_anchors,5+cfg.num_classes)
    output[:,:,:3] = torch.sigmoid(output[:,:,:3])
    output[:,:,3:5] = torch.exp(output[:,:,3:5])
    output[:,:,5:] = torch.sigmoid(output[:,:,5:])
    delta_pred_batch = output[:,:,1:5]
    all_grid_xywh = generate_all_anchors(torch.FloatTensor(cfg.anchors)/grid_size, H, W)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    result = []
    for b in range (B):
        box_pred = box_transform_inv(all_grid_xywh, output[b,:,1:5])
        box_pred = xywh2xxyy(box_pred)
        batch_result = {}
        for c in range (cfg.num_classes):
            probs = output[b,:,0] * output[b,:,5+c]
            keep  = Self_NMS(box_pred,probs,cfg.thresh)
            batch_result[c] = box_pred[keep] * grid_size
        result.append(batch_result)
    return result


def train(model, data_loader, test_loader, epochs):

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr = cfg.lr)

    for epoch in range (epochs):
        model.train()
        t_box_loss, t_iou_loss, t_class_loss = 0, 0, 0
        for images, groundTruth in tqdm(data_loader):
            images = images.cuda()
            groundTruth = {k:v.cuda() for k,v in groundTruth.items()}
    
            output = model(images)
    
            targets = build_target(output,(groundTruth["boxes"],groundTruth["labels"],groundTruth["num_objs"]))
    
            box_loss, iou_loss, class_loss = yolo_loss(output,targets)
            t_box_loss += box_loss.item()
            t_iou_loss += iou_loss.item()
            t_class_loss += class_loss.item()
            loss = box_loss + iou_loss + class_loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        print(box_loss, iou_loss, class_loss)
        
    model.eval()
    with torch.no_grad():
        precision = 0
        recall = 0
        batch = 0
        for images, groundTruth in tqdm(test_loader):
            images = images.cuda()
            groundTruth = {k:v.cuda() for k,v in groundTruth.items()}
            output = model(images)
            result = Full_NMS(output)
            tp = 0
            fp = 0
            t = 0
            for b in range (images.size(0)):
                image = images[b]
                num_objs = groundTruth["num_objs"][b]
                boxes = groundTruth["boxes"][b][:num_objs,:]
                labels = groundTruth["labels"][b][:num_objs]
                t += num_objs
                for c in range (cfg.num_classes):
                    label_boxes = boxes[torch.where(labels==c)]
                    pred_boxes = result[b][c]
                    if len (label_boxes) == 0:
                        fp += len(pred_boxes)
                        continue
                    ious,_ = torch.max(box_ious(pred_boxes,label_boxes),dim=-1)
                    tp += torch.sum(ious>0.95).item()
                    fp += pred_boxes.size(0) - tp
            precision += tp / (tp+fp)
            recall += tp / t
            batch += 1
        precision /= batch
        recall /= batch
        print(precision, recall)

model = YoloV2()

train(model, train_data_loader, val_data_set_loader, 20)
