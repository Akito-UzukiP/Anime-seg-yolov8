import ultralytics
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
import numpy as np
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from tqdm import tqdm
import parser

#
#函数列表：
# seg_any_mask_generate: 生成sam的mask
# yolo_mask_generate: 生成yolo的mask，同时返回bboxes，用于裁剪
# vote_mask_generate: 把所有跟YOLO_mask重合率大于threshold的SAM_mask进行或运算，最后返回
# seg_single: 用于单张图片的处理，返回最终的mask和裁剪后的图片
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
def yolo_mask_generate(yolo_model, image,expand_rate = 1.1):
    # 生成mask
    masks = yolo_model(image)
    bboxes = masks[0].boxes.xyxyn.detach().cpu().numpy()
    masks = masks[0].masks.data.data.cpu().numpy().sum(axis=0)
    # 按照bboxes的大小，对mask进行裁剪
    bboxes = np.concatenate((bboxes.min(axis=0)[:2],bboxes.max(axis=0)[2:]))
    return masks,bboxes

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
    #将final_mask里的小洞填上
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

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

def seg_single(seg_model, yolo_model, image, threshold=0.5):

    YOLO_masks, bboxes = yolo_mask_generate(yolo_model, image)
    # 2. 裁剪
    image = image[int(bboxes[1]*image.shape[0]):int(bboxes[3]*image.shape[0]),int(bboxes[0]*image.shape[1]):int(bboxes[2]*image.shape[1])]
    YOLO_masks = YOLO_masks[int(bboxes[1]*YOLO_masks.shape[0]):int(bboxes[3]*YOLO_masks.shape[0]),int(bboxes[0]*YOLO_masks.shape[1]):int(bboxes[2]*YOLO_masks.shape[1])]
    SAM_masks = segany_mask_generate(seg_model, image)
    # 3. 投票机制
    final_mask = vote_mask_generate(SAM_masks, YOLO_masks, threshold)
    return final_mask, image
def pipeline(seg_model, yolo_model, image_path, threshold=0.8):
    #用tqdm显示进度
    pbar = tqdm(os.listdir(image_path))
    for image_name in pbar:
        pbar.set_description("Processing %s" % image_name)
        image_path_ = os.path.join(image_path,image_name)
        image = Image.open(image_path_).convert("RGB")
        final_mask, image= seg_single(seg_model, yolo_model, np.array(image)[:,:,::-1], threshold)
        image = Image.fromarray(image)
        image_out = np.array(image.convert("RGBA"))
        #RGBA转成BGRA
        image_out[:,:,3] = (final_mask)*255
        #根据bboxes进行裁剪
        cv2.imwrite(os.path.join(image_path,image_name[:-4]+'_mask.png'),image_out)

if __name__ == '__main__':
    sam = sam_model_registry["vit_l"](checkpoint="./segany/sam_vit_l_0b3195.pth").to("cuda")
    sam_model_generator = SamAutomaticMaskGenerator(sam,
                                                pred_iou_thresh=0.6,
                                                stability_score_thresh=0.8,
                                                crop_n_points_downscale_factor=1,
                                                crop_n_layers=1)
    yolo_model = YOLO('./yolo/last.pt')
    path = r"D:\pyprojs\Anime-seg-yolov8\samples"
    pipeline( sam_model_generator, yolo_model, path)