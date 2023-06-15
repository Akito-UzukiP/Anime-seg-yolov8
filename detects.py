import ultralytics
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import numpy as np
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

#TODO: 1. SAM的mask生成器 2. yolov8的seg标注 3.投票机制
# 1. SAM的mask生成器
def segany_mask_generate(seg_model, image):
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    # 生成mask
    masks = seg_model.generate(image)
    return_masks = []
    for i, mask in enumerate(masks):
        return_masks.append(mask['segmentation'].astype(np.uint8))
    return np.array(return_masks)
# 2. yolov8的seg标注
def yolo_mask_generate(yolo_model, image):
    if isinstance(image, str):
        image = cv2.imread(image)
    if isinstance(image, Image.Image):
        image = np.array(image)
    # 生成mask
    masks = yolo_model(image)
    return masks[0].masks.data.data.cpu().numpy().sum(axis=0)

# 3. 投票机制，这里由于只有一类，所以直接融合所有YOLO_masks，然后和SAM_masks做投票，如果SAM_masks的重合率大于threshold就保留SAM_masks，否则扔掉，最后融合保留的SAM_masks并返回
def vote_mask_generate(SAM_masks,YOLO_masks,threshold=0.8):
    scores = []
    #把yolo_masks给resize到和SAM_masks一样的大小
    YOLO_masks = cv2.resize(YOLO_masks, (SAM_masks[0].shape[1], SAM_masks[0].shape[0]))
    final_mask = np.zeros_like(SAM_masks[0])
    for i, mask in enumerate(SAM_masks):
        score = np.sum(YOLO_masks*mask/mask.sum())
        scores.append(score)
        if score > threshold:
            #或运算
            final_mask = cv2.bitwise_or(final_mask,mask)
        #print(score)
    return final_mask
    
def plot_sam_mask(masks):
    color = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
    background = np.zeros((masks[0].shape[0],masks[0].shape[1],3),dtype=np.uint8)
    for i, mask in enumerate(masks):
        # 产生一个三通道的随机颜色 
        temp = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]),axis=2)
        temp = temp*color[i]
        background = background + temp
    return background