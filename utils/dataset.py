import os
import glob
import torch
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset


def _resolve_stare_gt(gt_dir: str, base_name: str) -> str:
    """STARE 真值文件名兼容解析（GTFIX）：imXXXX_ah.png → imXXXX.png / _vk 等依次探测。"""
    base = base_name
    for suf in ("_ah", "_vk", "_av"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    for name_suf in ("_ah", "_vk", "", "_av"):
        for ext in (".png", ".ppm", ".gif", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"):
            cand = os.path.join(gt_dir, f"{base}{name_suf}{ext}")
            if os.path.exists(cand):
                return cand
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(gt_dir, f"{base}*")))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"STARE 真值未找到：{gt_dir}/{base}[_ah|_vk]<ext>；目录示例：{sorted(os.listdir(gt_dir))[:6]}")


class DRIVEDataset(Dataset):
    def __init__(self, data_dir, mode='training', use_maskvsc=False, crop_size=128):
        self.data_dir = data_dir
        self.mode = mode
        self.use_maskvsc = use_maskvsc
        self.crop_size = crop_size

        self.img_dir = os.path.join(data_dir, mode, 'images')
        self.gt_dir = os.path.join(data_dir, mode, '1st_manual')
        self.fov_dir = os.path.join(data_dir, mode, 'mask')

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.tif")))
        if not self.img_files:
            raise FileNotFoundError(f"Cannot find .tif images in {self.img_dir}")
        aug_info = f"{self.crop_size}×{self.crop_size} crop + Aug" if self.mode == 'training' else "no crop"
        print(f"DRIVE {mode.upper()} Set loaded: {len(self.img_files)} images (RGB + FOV + {aug_info})"
              + (" [MaskVSC: 512×512, H+V-flip]" if use_maskvsc else ""))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        base_name = os.path.basename(img_path).split('_')[0]
        gt_path = os.path.join(self.gt_dir, f"{base_name}_manual1.gif")
        fov_path = os.path.join(self.fov_dir, f"{base_name}_{self.mode}_mask.gif")

        # 读取原始图像（保持原始分辨率）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')
        fov = Image.open(fov_path).convert('L')

        if self.use_maskvsc:
            # MaskVSC: resize 到 512×512, 水平翻转 + 垂直翻转, 无随机裁剪
            image = image.resize((512, 512), Image.BILINEAR)
            mask = mask.resize((512, 512), Image.NEAREST)
            fov = fov.resize((512, 512), Image.NEAREST)
            if self.mode == 'training':
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                    fov = TF.vflip(fov)
        else:
            # 动态 Padding：使图像长宽为 32 的整数倍
            w, h = image.size
            pad_w = (32 - w % 32) % 32
            pad_h = (32 - h % 32) % 32
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0, padding_mode='constant')
            mask = TF.pad(mask, padding, fill=0, padding_mode='constant')
            fov = TF.pad(fov, padding, fill=0, padding_mode='constant')

            # 数据增强（仅在训练模式）
            if self.mode == 'training':
                # 颜色抖动
                image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                    fov = TF.vflip(fov)
                # 随机裁剪
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                fov = TF.crop(fov, i, j, h, w)

        # 归一化：使用 TF 标准转换和 FRD-Net 的均值方差
        image = TF.to_tensor(image)  # 自动转为 [0, 1] 且 CHW
        if os.environ.get("PHRNET_NO_NORM") == "1":
            pass  # 官方权重口径：仅 to_tensor 的 [0,1]（M2U-Net 作者 dataset.py 为 center_crop+to_tensor，无归一化）
        else:
            image = TF.normalize(image, mean=[0.215, 0.215, 0.215], std=[0.151, 0.151, 0.151])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        fov = TF.to_tensor(fov)
        fov = (fov > 0.5).float()

        return image, mask, fov


class CHASEDB1Dataset(Dataset):
    def __init__(self, data_dir, mode='training', use_maskvsc=False, crop_size=128):
        self.data_dir = data_dir
        self.mode = mode
        self.use_maskvsc = use_maskvsc
        self.crop_size = crop_size

        # 获取所有原图，CHASE_DB1 共有 28 张图
        all_img_files = sorted(glob.glob(os.path.join(self.data_dir, "*.jpg")))
        if not all_img_files:
            raise FileNotFoundError(f"Cannot find .jpg images in {self.data_dir}")

        # 所有 mode 都使用完整的数据集，只是处理方式不同
        # 按照业界标准划分：前 20 张为训练集，后 8 张为测试/验证集
        # 但为了支持 k 折交叉验证，我们在所有 mode 下都保留完整的文件列表
        self.img_files = all_img_files
        # 记录原始划分，用于后续可能的使用
        self.train_files = all_img_files[:20]
        self.test_files = all_img_files[20:]
        if mode not in ['training', 'validation', 'test']:
            raise ValueError("mode 必须是 'training', 'validation' 或 'test'")

        aug_info = f"{self.crop_size}×{self.crop_size} crop + Aug" if self.mode == 'training' else "no crop"
        print(f"CHASE_DB1 {mode.upper()} Set loaded: {len(self.img_files)} images (RGB + Auto FOV + {aug_info})"
              + (" [MaskVSC: 512×512, H+V-flip]" if use_maskvsc else ""))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # 金标准使用第一位专家的标注 (1stHO)
        gt_path = img_path.replace('.jpg', '_1stHO.png')

        # 读取原始图像（保持原始分辨率）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')

        # 自动生成 FOV: 提取灰度图中亮度大于 10 的区域作为眼底有效视野
        gray_img = np.array(image.convert('L'))
        fov = (gray_img > 10).astype(np.uint8) * 255
        fov = Image.fromarray(fov).convert('L')

        if self.use_maskvsc:
            image = image.resize((512, 512), Image.BILINEAR)
            mask = mask.resize((512, 512), Image.NEAREST)
            fov = fov.resize((512, 512), Image.NEAREST)
            if self.mode == 'training':
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
        else:
            # 动态 Padding：使图像长宽为 32 的整数倍
            w, h = image.size
            pad_w = (32 - w % 32) % 32
            pad_h = (32 - h % 32) % 32
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0, padding_mode='constant')
            mask = TF.pad(mask, padding, fill=0, padding_mode='constant')
            fov = TF.pad(fov, padding, fill=0, padding_mode='constant')

            # 数据增强（仅在训练模式）
            if self.mode == 'training':
                # 颜色抖动
                image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                    fov = TF.vflip(fov)
                # 随机裁剪
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                fov = TF.crop(fov, i, j, h, w)

        # 归一化：使用 TF 标准转换和 FRD-Net 的均值方差
        image = TF.to_tensor(image)  # 自动转为 [0, 1] 且 CHW
        if os.environ.get("PHRNET_NO_NORM") == "1":
            pass  # 官方权重口径：仅 to_tensor 的 [0,1]（M2U-Net 作者 dataset.py 为 center_crop+to_tensor，无归一化）
        else:
            image = TF.normalize(image, mean=[0.215, 0.215, 0.215], std=[0.151, 0.151, 0.151])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        fov = TF.to_tensor(fov)
        fov = (fov > 0.5).float()

        return image, mask, fov


class STAREDataset(Dataset):
    def __init__(self, data_dir, mode='training', use_maskvsc=False, crop_size=128):
        self.data_dir = data_dir
        self.mode = mode
        self.use_maskvsc = use_maskvsc
        self.crop_size = crop_size

        # 直接读取预处理好的干净文件夹
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.gt_dir = os.path.join(self.data_dir, 'masks')

        all_img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        if not all_img_files:
            raise FileNotFoundError(f"Cannot find .png images in {self.img_dir}")

        # 所有 mode 都使用完整的数据集，只是处理方式不同
        # STARE 业界标准：前 10 张训练，后 10 张测试
        # 但为了支持 k 折交叉验证，我们在所有 mode 下都保留完整的文件列表
        self.img_files = all_img_files
        # 记录原始划分，用于后续可能的使用
        self.train_files = all_img_files[:10]
        self.test_files = all_img_files[10:]
        if mode not in ['training', 'validation', 'test']:
            raise ValueError("mode 必须是 'training', 'validation' 或 'test'")

        aug_info = f"{self.crop_size}×{self.crop_size} crop + Aug" if self.mode == 'training' else "no crop"
        print(f"STARE {mode.upper()} Set loaded: {len(self.img_files)} images (RGB + Auto FOV + {aug_info})"
              + (" [MaskVSC: 512×512, H+V-flip]" if use_maskvsc else ""))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        # 解析预处理后的文件名: im0001.png -> im0001_ah.png
        base_name = os.path.basename(img_path).replace('.png', '')
        gt_path = _resolve_stare_gt(self.gt_dir, base_name)  # GTFIX

        # 读取原始图像（保持原始分辨率）
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')

        # 自动生成 FOV
        gray_img = np.array(image.convert('L'))
        fov = (gray_img > 10).astype(np.uint8) * 255
        fov = Image.fromarray(fov).convert('L')

        if self.use_maskvsc:
            image = image.resize((512, 512), Image.BILINEAR)
            mask = mask.resize((512, 512), Image.NEAREST)
            fov = fov.resize((512, 512), Image.NEAREST)
            if self.mode == 'training':
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
        else:
            # 动态 Padding：使图像长宽为 32 的整数倍
            w, h = image.size
            pad_w = (32 - w % 32) % 32
            pad_h = (32 - h % 32) % 32
            padding = (0, 0, pad_w, pad_h)  # left, top, right, bottom
            image = TF.pad(image, padding, fill=0, padding_mode='constant')
            mask = TF.pad(mask, padding, fill=0, padding_mode='constant')
            fov = TF.pad(fov, padding, fill=0, padding_mode='constant')

            # 数据增强（仅在训练模式）
            if self.mode == 'training':
                # 颜色抖动
                image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
                image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
                if random.random() > 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                    fov = TF.hflip(fov)
                if random.random() > 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                    fov = TF.vflip(fov)
                # 随机裁剪
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                fov = TF.crop(fov, i, j, h, w)

        # 归一化：使用 TF 标准转换和 FRD-Net 的均值方差
        image = TF.to_tensor(image)  # 自动转为 [0, 1] 且 CHW
        if os.environ.get("PHRNET_NO_NORM") == "1":
            pass  # 官方权重口径：仅 to_tensor 的 [0,1]（M2U-Net 作者 dataset.py 为 center_crop+to_tensor，无归一化）
        else:
            image = TF.normalize(image, mean=[0.215, 0.215, 0.215], std=[0.151, 0.151, 0.151])
        mask = TF.to_tensor(mask)
        mask = (mask > 0.5).float()
        fov = TF.to_tensor(fov)
        fov = (fov > 0.5).float()

        return image, mask, fov
