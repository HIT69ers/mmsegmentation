import cv2
import torch
import numpy as np

from mmengine.registry import init_default_scope
from mmseg.datasets import NYUV2Dataset

init_default_scope('mmseg')

data_root = 'data/nyuv2'
data_prefix = dict(img_path='images/train', depth_map_path='annotations/train')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.),
    dict(type='PackSegInputs')
]

dataset = NYUV2Dataset(data_root=data_root, data_prefix=data_prefix, test_mode=False, pipeline=train_pipeline, reduce_zero_label=False)

print(len(dataset))
print(dataset.get_data_info(0))
print(dataset.metainfo)
print(dataset[0])
print(dataset[0]['data_samples'].gt_sem_seg.data)

img_numpy = dataset[0]['inputs'].cpu().numpy()

# 2. 调整维度顺序：从 (C, H, W) 转换为 (H, W, C)
# 假设你的 tensor 形状是 [3, Height, Width]
img_numpy = np.transpose(img_numpy, (1, 2, 0))

# 3. 颜色空间转换：从 RGB 转换为 BGR (仅针对 3 通道彩色图像)
if img_numpy.shape[-1] == 3:
    # img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    img_bgr = img_numpy
else:
    # 如果是单通道灰度图，则不需要转换颜色空间
    img_bgr = img_numpy 

# 4. 使用 OpenCV 可视化
cv2.imshow('Tensor Image', img_bgr)


# 1. 定义 NYU 数据集的颜色映射表 (Colormap)
# 这里直接使用你提供的映射数组
nyu_colormap = np.load('nyucmap.npy')

# 2. 准备gt
label_tensor = dataset[0]['data_samples'].gt_sem_seg.data

def visualize_segmentation_mask(tensor_mask, colormap):
    """
    将单通道的类别索引 Tensor 转换为 RGB/BGR 可视化图像
    """
    # 将 Tensor 转到 CPU 并转为 Numpy 数组
    # squeeze() 会去掉形状为 1 的维度 (例如从 (1, H, W) 变成 (H, W))
    mask_np = tensor_mask.squeeze().cpu().numpy()
    
    # 3. 将 255 视为 0 (处理忽略的标签或背景)
    mask_np[mask_np == 255] = 0
    
    # 4. 利用 NumPy 高级索引，直接将二维索引数组映射为三维彩色图像
    # 这步执行后，color_image 的形状将变为 (H, W, 3)
    color_image_rgb = colormap[mask_np]
    
    # 5. OpenCV 默认使用 BGR 格式，而映射表通常是 RGB 格式，需要转换
    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
    
    return color_image_bgr
    # return color_image_rgb

# 运行可视化函数
result_img = visualize_segmentation_mask(label_tensor, nyu_colormap)

# 使用 OpenCV 显示
cv2.imshow('NYU Segmentation Mask', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()