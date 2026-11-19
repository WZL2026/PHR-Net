import torch
import numpy as np
from scipy.ndimage import morphology
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def compute_metrics_accumulate(pred, target, threshold=0.5):
    """用于训练/验证循环中快速累加 TP, FP, FN"""
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred_mask = (pred > threshold).float()
    target_mask = (target > 0.5).float()

    tp = (pred_mask * target_mask).sum().item()
    fp = (pred_mask * (1 - target_mask)).sum().item()
    fn = ((1 - pred_mask) * target_mask).sum().item()
    return tp, fp, fn


def compute_cldice(pred, gt, epsilon=1e-6):
    """
    clDice (centerline Dice) for tubular structure segmentation.
    pred, gt: binary masks (H, W) as float32 or bool
    """
    p = (pred > 0.5).astype(bool)
    g = (gt > 0.5).astype(bool)

    # Skeletonize
    skel_p = skeletonize(p)
    skel_g = skeletonize(g)

    # Tprec: fraction of predicted skeleton that falls within ground truth
    tprec = (skel_p & g).sum() / (skel_p.sum() + epsilon)
    # Tsens: fraction of ground truth skeleton that falls within prediction
    tsens = (skel_g & p).sum() / (skel_g.sum() + epsilon)

    return 2.0 * tprec * tsens / (tprec + tsens + epsilon)
    """科研级 Dice"""
    p = (pred > 0.5).astype(bool)
    g = (gt > 0.5).astype(bool)
    intersection = (p & g).sum()
    union = p.sum() + g.sum()
    return 2.0 * intersection / union if union > 0 else 1.0


def get_surface_points(mask):
    """计算表面点用于 HD95/ASD"""
    if not np.any(mask): return np.array([])
    eroded = morphology.binary_erosion(mask)
    surface = mask ^ eroded
    return np.argwhere(surface)


def compute_hd95(pred, gt):
    """95% Hausdorff Distance"""
    p_points = get_surface_points(pred > 0.5)
    g_points = get_surface_points(gt > 0.5)
    if len(g_points) == 0: return 0.0 if len(p_points) == 0 else np.nan
    if len(p_points) == 0: return np.nan

    dist_p2g = cKDTree(g_points).query(p_points)[0]
    dist_g2p = cKDTree(p_points).query(g_points)[0]
    return np.percentile(np.concatenate([dist_p2g, dist_g2p]), 95)


def sliding_window_inference(model, image, window_size=(128, 128), stride=(64, 64), device='cuda', is_maskvsc=False):
    """
    滑动窗口推理函数
    Args:
        model: 模型
        image: 输入图像 (1, 3, H, W)
        window_size: 窗口大小
        stride: 滑动步长
        device: 设备
        is_maskvsc: 是否为 MaskVSC 模型（输出已经过 Sigmoid）
    Returns:
        全图预测结果 (H, W)
    """
    model.eval()
    H, W = image.shape[2], image.shape[3]
    output = torch.zeros((1, 1, H, W), device=device)
    count = torch.zeros((1, 1, H, W), device=device)
    
    # 计算窗口数量
    num_windows_h = (H - window_size[0]) // stride[0] + 1
    num_windows_w = (W - window_size[1]) // stride[1] + 1
    
    # 处理边界情况
    if (H - window_size[0]) % stride[0] != 0:
        num_windows_h += 1
    if (W - window_size[1]) % stride[1] != 0:
        num_windows_w += 1
    
    with torch.no_grad():
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                # 计算当前窗口的起始位置
                start_h = i * stride[0]
                start_w = j * stride[1]
                end_h = min(start_h + window_size[0], H)
                end_w = min(start_w + window_size[1], W)
                
                # 调整窗口大小以适应边界
                current_window_size = (end_h - start_h, end_w - start_w)
                
                # 提取窗口
                window = image[:, :, start_h:end_h, start_w:end_w]
                
                # 4-fold TTA 预测
                # 1. 准备 4 种翻转形态的输入
                patch_orig = window
                patch_h = torch.flip(window, dims=[3])       # 水平翻转
                patch_v = torch.flip(window, dims=[2])       # 垂直翻转
                patch_hv = torch.flip(window, dims=[2, 3])   # 对角翻转
                
                # 2. 分别进行前向传播 (统一处理深层监督的返回格式)
                def get_pred(x):
                    out = model(x)
                    # 如果模型返回的是 tuple (如 PHR_Net 训练/推理行为可能不同，稳妥起见取第一个)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    return out
                
                pred_orig = get_pred(patch_orig)
                pred_h = get_pred(patch_h)
                pred_v = get_pred(patch_v)
                pred_hv = get_pred(patch_hv)
                
                # 3. 将翻转后的预测结果反向翻转回来
                pred_h_inv = torch.flip(pred_h, dims=[3])
                pred_v_inv = torch.flip(pred_v, dims=[2])
                pred_hv_inv = torch.flip(pred_hv, dims=[2, 3])
                
                # 4. 求 4 个结果的平均值
                pred = (pred_orig + pred_h_inv + pred_v_inv + pred_hv_inv) / 4.0
                
                # 将预测结果放回对应位置
                output[:, :, start_h:end_h, start_w:end_w] += pred
                count[:, :, start_h:end_h, start_w:end_w] += 1
    
    # 平均值融合
    output = output / count
    
    # MaskVSC 输出已经经过 Sigmoid，不需要再处理
    if not is_maskvsc:
        output = torch.sigmoid(output)

    return output.squeeze().cpu().numpy()


def direct_tta_inference(model, image, device='cuda', is_maskvsc=False):
    """
    直接全图 TTA 推理 — 绕过滑动窗口的补丁循环开销。
    适用场景: 输入已 resize 到模型训练尺寸 (如 512×512)。

    TTA 策略: 原图 + 水平翻转 + 垂直翻转 + 对角翻转 → 平均融合。
    相比 sliding_window_inference 省去窗口分割/计数图/除法开销。
    """
    model.eval()
    with torch.no_grad():
        # 4-fold TTA
        def _forward(x):
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        p_orig = _forward(image)
        p_h = torch.flip(_forward(torch.flip(image, dims=[3])), dims=[3])
        p_v = torch.flip(_forward(torch.flip(image, dims=[2])), dims=[2])
        p_hv = torch.flip(_forward(torch.flip(image, dims=[2, 3])), dims=[2, 3])

        fused = (p_orig + p_h + p_v + p_hv) / 4.0

        if not is_maskvsc:
            fused = torch.sigmoid(fused)

    return fused.squeeze().cpu().numpy()