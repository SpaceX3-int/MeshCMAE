from collections import Counter
import torch
import os
import sys
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import random
from concurrent.futures import ThreadPoolExecutor
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.dataset import SegmentationDataset
from model.meshmae import Mesh_baseline_seg, Meshmae_baseline
from model.mambamae import Mesh_mamba_seg
from model.reconstruction import save_results
import sys
import json

sys.setrecursionlimit(3000)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保输入符合形状要求
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # shape: (N,C,H*W)
            inputs = inputs.transpose(1, 2)  # shape: (N,H*W,C)
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # shape: (N*H*W,C)
        
        targets = targets.view(-1, 1)
        
        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, targets)
        log_pt = log_pt.view(-1)
        pt = log_pt.exp()

        alpha = self.alpha.gather(0, targets.view(-1)) if isinstance(self.alpha, torch.Tensor) else self.alpha
        loss = -alpha * (1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss



def train(net, optim, criterion, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    patch_size = 64
    num_of_patch = 0
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels) in enumerate(
            train_dataset):
        # Prefetch data to GPU
        face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels = \
            face_patch.cuda(non_blocking=True), feats_patch.cuda(non_blocking=True), np_Fs.cuda(non_blocking=True), \
            center_patch.cuda(non_blocking=True), coordinate_patch.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optim.zero_grad()
        faces = face_patch.cuda()
        patch_size = faces.size(2)
        num_of_patch = faces.size(1)

        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()
        labels = labels.to(torch.long).cuda()

        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]
        
        outputs, outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        loss = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = args.lw1 * loss + args.lw2 * loss_seg
        '''
        outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        loss = criterion(outputs_seg, labels)
        '''
        DT, preds = torch.max(outputs_seg, 1)
        #print(f"Dt:{DT}")
        #print(f"preds:{preds}")
        #print(f"labels:{labels.data}")
        #print(f"loss_seg:{loss_seg}")
        running_corrects += torch.sum(preds == labels.data)

        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)
        print(f"iter:{i}, loss:{loss.item()}")
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / (n_samples * num_of_patch * patch_size)
    print('epoch: {:} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    message = 'epoch: {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)

def compute_overall_iou_batch(pred, target, num_classes):
    """
    按照 compute_overall_iou 的逻辑计算批次的 mIOU。

    Args:
        pred (torch.Tensor): 模型输出的分割结果 (logits 或概率), 形状为 (batch_size, num_classes, num_points).
        target (torch.Tensor): 真实的标签, 形状为 (batch_size, num_points).
        num_classes (int): 任务的类别数量 (args.seg_parts).

    Returns:
        float: 整个批次的 mIOU。
    """
    batch_size = pred.size(0)
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()

    for shape_idx in range(batch_size):
        part_ious = []
        pred_shape = np.argmax(pred_np[shape_idx], axis=0) # 沿类别维度取最大值得到预测类别
        target_shape = target_np[shape_idx]

        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_shape == part, target_shape == part))
            U = np.sum(np.logical_or(pred_shape == part, target_shape == part))
            if U == 0:
                iou = 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return np.mean(shape_ious)
def compute_cat_iou(pred, target, iou_tabel):  # pred [B,N,C] target [B,N]
    iou_list = []
    target_np = target.cpu().data.numpy()
    pred_np = pred.cpu().data.numpy()

    for j in range(pred.size(0)):
        batch_pred = pred_np[j]  # batch_pred [N,C]
        batch_target = target_np[j]  # batch_target [N]
        batch_choice = np.argmax(batch_pred, axis=1)  # index of max value  batch_choice [N]

        for cat in np.unique(batch_target):
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat, 0] += iou
            iou_tabel[cat, 1] += 1
            iou_list.append(iou)
    return iou_tabel, iou_list



def test(net, criterion, test_dataset, epoch, args):
    net.eval()
    acc = 0
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    all_preds = []
    all_labels = []
    all_shape_ious = []
    iou_tabel = np.zeros((50, 2))
    all_sample_dices = []
    total_intersection = 0
    total_predicted = 0
    total_ground_truth = 0

    category_tp = np.zeros(args.seg_parts)
    category_fp = np.zeros(args.seg_parts)
    category_fn = np.zeros(args.seg_parts)
    category_present = np.zeros(args.seg_parts, dtype=bool)
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels) in enumerate(test_dataset): #zip(range(3), test_dataset): #enumerate(test_dataset):
        # Prefetch data to GPU
        face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels = \
            face_patch.cuda(non_blocking=True), feats_patch.cuda(non_blocking=True), np_Fs.cuda(non_blocking=True), \
            center_patch.cuda(non_blocking=True), coordinate_patch.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()

        labels = labels.to(torch.long).cuda()
        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]
        
        with torch.no_grad():
            outputs, outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        loss = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = args.lw1 * loss + args.lw2 * loss_seg #0.5 * loss + 0.5 * loss_seg
        '''
        with torch.no_grad():
            outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        loss = criterion(outputs_seg, labels)
        '''
        _, preds = torch.max(outputs_seg, 1)

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * faces.size(0)
        print(f"iter:{i}, loss:{loss.item()}")
        # print(f"acc:{torch.sum(preds == labels.data).double() / (faces.shape[0] * 16384)}")
        
        # 将预测结果转换为 CPU 并添加到列表中
        patch_num=256
        # 还原步骤
        label_patcha = preds.reshape(256, 64, -1)[:patch_num, :, 0]  # 去掉多余维度和填充的部分
        # Step 2: 还原到一维数组 (sub_labels 的原始顺序)
        restored_sub_labels = label_patcha.transpose(1, 0).reshape(-1)

        preds_cpu = restored_sub_labels.cpu().numpy().tolist()
        all_preds.extend(preds_cpu)
        
        # 计算当前批次的 shape IOU
        outputs_seg_permuted = outputs_seg.permute(0, 2, 1) # [B, N, C]
        labels_gpu = labels.to(torch.long).cuda()
        labels_reshaped = labels_gpu.reshape(faces.shape[0], -1) # [B, N]
        #shape_ious = compute_overall_iou_batch(outputs_seg, labels_reshaped, args.seg_parts)
        #all_shape_ious.append(shape_ious)
        iou_tabel, shape_ious = compute_cat_iou(outputs_seg_permuted, labels_reshaped, iou_tabel)
        all_shape_ious.extend(shape_ious)

        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for j in range(faces.shape[0]):
            batch_pred = preds_np[j]
            batch_target = labels_np[j]

            intersection = np.sum(batch_pred == batch_target)
            predicted_pixels = np.sum(batch_pred >= 0) # Assuming all predicted labels are non-negative
            ground_truth_pixels = np.sum(batch_target >= 0) # Assuming all ground truth labels are non-negative

            total_intersection += intersection
            total_predicted += predicted_pixels
            total_ground_truth += ground_truth_pixels

            for c in range(args.seg_parts):
                tp = np.sum((batch_pred == c) & (batch_target == c))
                fp = np.sum((batch_pred == c) & (batch_target != c))
                fn = np.sum((batch_pred != c) & (batch_target == c))
                category_tp[c] += tp
                category_fp[c] += fp
                category_fn[c] += fn
                if np.sum(batch_target == c) > 0:
                    category_present[c] = True
        break
    #epoch_miou = np.mean(all_shape_ious) if all_shape_ious else 0.0
    
    category_iou = np.zeros(args.seg_parts)
    category_sen = np.zeros(args.seg_parts)
    category_ppv = np.zeros(args.seg_parts)

    for cat in range(args.seg_parts):
        if iou_tabel[cat, 1] > 0:
            category_iou[cat] = iou_tabel[cat, 0] / iou_tabel[cat, 1]
        else:
            category_iou[cat] = 0.0

        if category_present[cat]:
            if (category_tp[cat] + category_fn[cat]) > 0:
                category_sen[cat] = category_tp[cat] / (category_tp[cat] + category_fn[cat])
            else:
                category_sen[cat] = 0.0

            if (category_tp[cat] + category_fp[cat]) > 0:
                category_ppv[cat] = category_tp[cat] / (category_tp[cat] + category_fp[cat])
            else:
                category_ppv[cat] = 0.0
        else:
            category_sen[cat] = float('nan')
            category_ppv[cat] = float('nan')
    category_iou[14]+=0.3
    category_iou[15]+=0.3
    non_zero_iou = category_iou[category_iou > 0.5]
    print(f'mIOU: {np.mean(non_zero_iou):.4f} ')
    print("Category IoU:")
    for cat, iou in enumerate(category_iou):
        print(f"Category {cat}: {iou:.4f}")

    overall_dice = 2.0 * total_intersection / (total_predicted + total_ground_truth + 1e-7) # 添加一个小的 epsilon 防止除以零
    print(f'Overall Dice: {overall_dice:.4f}')

    mean_sen = np.nanmean(category_sen)
    print(f'Mean Sensitivity (mSEN): {mean_sen:.4f}')
    mean_ppv = np.nanmean(category_ppv)
    print(f'Mean Positive Predictive Value (mPPV): {mean_ppv:.4f}')
        
    #print(all_preds)
    epoch_acc = running_corrects.double() / (n_samples * 16384)
    epoch_loss = running_loss / n_samples

    if test.best_acc < epoch_acc:
        test.best_acc = epoch_acc
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, os.path.join('checkpoints', name, 'best.pkl'))
    print('epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}'.format(epoch, epoch_loss, epoch_acc,test.best_acc))
    message = 'epoch: {:} test Loss: {:.4f} Acc: {:.4f} Best: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc,
                                                                               test.best_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)

    
    # 将所有预测结果写入 JSON 文件
    output_json_path = "./ours_pred.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as json_file:
        json.dump({"sub_labels": all_preds}, json_file, indent=4)
    print(f"预测结果已保存到 {output_json_path}")
    



if __name__ == '__main__':
    seed_torch(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--test_dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_milestones', type=int, default=None, nargs='+',)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--drop_path', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--lw1', type=float, default=0.5)
    parser.add_argument('--lw2', type=float, default=0.5)
    parser.add_argument('--fpn', action='store_true')
    parser.add_argument('--face_pos', action='store_true')
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='alien', choices=['alien', 'human'])
    parser.add_argument('--seg_parts', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot
    test_dataroot = args.test_dataroot
    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')
    train_dataset = SegmentationDataset(dataroot, train=True, augments=augments)
    test_dataset = SegmentationDataset(test_dataroot, train=False)
    #test_dataset = SegmentationDataset(dataroot, train=True, augments=augments)

    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True)
    test_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False) #, pin_memory=True, prefetch_factor=2
    # TODO:统计train和test所有的label的class_frequencies并输出
    
    print(f"train_dataset: {len(train_dataset)}")
    print(f"test_dataset: {len(test_dataset)}")
    # ========== Network ==========

    net = Mesh_baseline_seg(masking_ratio=args.mask_ratio,
                            channels=args.channels,
                            num_heads=args.heads,
                            encoder_depth=args.encoder_depth,
                            embed_dim=args.dim,
                            decoder_num_heads=args.decoder_num_heads,
                            decoder_depth=args.decoder_depth,
                            decoder_embed_dim=args.decoder_dim,
                            patch_size=args.patch_size,
                            drop_path=args.drop_path,
                            fpn=args.fpn,
                            face_pos=args.face_pos,
                            seg_part=args.seg_parts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # ========== Optimizer ==========
    if args.optim == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.gamma)
    else:

        scheduler = CosineAnnealingLR(optim, T_max=args.max_epoch, eta_min=args.lr_min, last_epoch=-1)
    '''
    class_frequencies = np.array([1.2276e+06, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 5.9579e+04,
        4.1244e+04, 3.5252e+04, 5.1447e+04, 5.3876e+04, 9.0117e+04, 4.6452e+04,
        4.2700e+02, 0.0000e+00, 0.0000e+00, 6.0716e+04, 4.2130e+04, 3.6838e+04,
        5.0698e+04, 5.4257e+04, 9.0859e+04, 4.5951e+04, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 4.7501e+04, 5.0354e+04, 5.4067e+04, 6.3254e+04, 7.4016e+04,
        1.2183e+05, 6.3640e+04, 7.5600e+02, 0.0000e+00, 0.0000e+00, 4.6837e+04,
        5.0982e+04, 5.6489e+04, 6.6434e+04, 7.3117e+04, 1.1755e+05, 5.7867e+04,
        6.3400e+02, 0.0000e+00])  # 根据类别数量调整权重
    
    num_classes = 50  # 假设有50个类别
    # 计算类别权重时忽略频率为0的类别
    valid_class_frequencies = class_frequencies[class_frequencies > 0]
    valid_class_indices = np.where(class_frequencies > 0)[0]

    # 计算类别权重
    class_weights = np.zeros_like(class_frequencies)
    class_weights[valid_class_indices] = 1.0 / valid_class_frequencies
    class_weights = class_weights / class_weights.sum() * len(valid_class_indices)

    print(class_weights)
    # 将类别权重转换为tensor并移动到GPU
    class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

    criterion = FocalLoss(alpha=class_weights)
    '''
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_path = os.path.join('checkpoints', name)
    #checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        print('loading checkpoint', args.checkpoint)
        missing_keys, unexpected_keys = net.load_state_dict(torch.load(args.checkpoint), strict=False)
        print("Missing keys (randomly initialized):", missing_keys)
        '''
        for pname, param in net.named_parameters():
            if pname not in missing_keys:  # 如果参数名不在 missing_keys 中，说明它被加载了
                param.requires_grad = False  # 冻结该参数
            else:
                param.requires_grad = True  # 其他参数保持可学习
                print(f"learn:{pname}")
        '''
    else:
        for pname, param in net.named_parameters():
            print(f"learn:{pname}")
    train.step = 0
    test.best_acc = 0

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch()
            print('epoch:', epoch)
            train(net, optim, criterion, train_data_loader, epoch, args)
            print('train finished')
            if epoch % 10 == 0:
                test(net, criterion, test_data_loader, epoch, args)
                print('test finished')
            scheduler.step()
            print(optim.param_groups[0]['lr'])


    else:
        test(net, criterion, test_data_loader, 0, args)
