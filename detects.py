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
import argparse

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
    original_image_size = image.shape[:2]
    # 如果图片太大，就resize到1920以下，最长边为1920
    if max(original_image_size) > 1920:
        scale = 1920 / max(original_image_size)
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    masks = seg_model.generate(image)
    return_masks = []
    for i, mask in enumerate(masks):
        #把masks给resize到原图大小
        return_masks.append(mask['segmentation'].astype(np.uint8))
    return np.array(return_masks)
# 2. yolov8的seg标注
def yolo_mask_generate(yolo_model, image):
    # 生成mask
    masks = yolo_model(image)
    if masks is None:
        return None
    masks = masks[0].masks.data.data.cpu().numpy().sum(axis=0)
    #获取数量
    return masks

# 3. 投票机制，这里由于只有一类，所以直接融合所有YOLO_masks，然后和SAM_masks做投票，如果SAM_masks的重合率大于threshold就保留SAM_masks，否则扔掉，最后融合保留的SAM_masks并返回
def vote_mask_generate(SAM_masks,YOLO_masks,threshold=0.8,num = 1,erode_iter=1,erode_kernel_size=2):
    scores = []
    #把yolo_masks给resize到和SAM_masks一样的大小
    zeros = np.zeros_like(SAM_masks[0])
    #zeros与所有的SAM_masks进行或运算，得到一个和SAM_masks一样大小的mask 然后取反，得到所有的未标注区域，然后与YOLO_masks相与，得到所有的未标注区域的YOLO_masks
    full_mask = np.zeros_like(SAM_masks[0])
    for i, mask in enumerate(SAM_masks):
        full_mask = cv2.bitwise_or(full_mask,mask)
    #对YOLO_masks先进行一些腐蚀
    kernel = np.ones((erode_kernel_size,erode_kernel_size),np.uint8)
    YOLO_masks = cv2.erode(YOLO_masks,kernel,iterations = erode_iter)
    YOLO_masks = cv2.resize(YOLO_masks, (SAM_masks[0].shape[1], SAM_masks[0].shape[0])).astype(np.uint8)
    not_labeled = cv2.bitwise_not(cv2.bitwise_or(zeros, full_mask))
    final_mask = np.zeros_like(SAM_masks[0])
    for i, mask in enumerate(SAM_masks):
        score = np.sum(YOLO_masks*mask/mask.sum())
        scores.append(score)
        if score > threshold:
            #或运算
            final_mask = cv2.bitwise_or(final_mask,mask)
        #print(score)
    not_labeled_yolo = cv2.bitwise_and(not_labeled, YOLO_masks)
    final_mask = cv2.bitwise_or(final_mask,not_labeled_yolo)
    return final_mask.astype(np.uint8)
    
def plot_sam_mask(masks):
    color = np.random.randint(0, 255, (len(masks), 3), dtype=np.uint8)
    background = np.zeros((masks[0].shape[0],masks[0].shape[1],3),dtype=np.uint8)
    for i, mask in enumerate(masks):
        # 产生一个三通道的随机颜色 
        temp = np.concatenate((mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]),axis=2)
        temp = temp*color[i]
        background = background + temp
    return background


def pipeline(seg_model, yolo_model, image_path, threshold=0.3):
    output_masks = []
    images = []
    output_images = []
    if os.path.isdir(image_path):
        for image_ in os.listdir(image_path):
            image = Image.open(os.path.join(image_path,image_)).convert("RGB")
            yolo_masks= yolo_mask_generate(yolo_model,image)
            if yolo_masks is None:
                continue
            sam_masks = segany_mask_generate(seg_model,image)
            voted_mask = vote_mask_generate(sam_masks,yolo_masks,threshold=threshold)
            output_masks.append(voted_mask)
            images.append(image)
        if len(output_masks) == 0:
            return None
    else:
        image = Image.open(image_path).convert("RGB")
        yolo_masks= yolo_mask_generate(yolo_model,image)
        if yolo_masks is None:
            return None
        sam_masks = segany_mask_generate(seg_model,image)
        voted_mask = vote_mask_generate(sam_masks,yolo_masks,threshold=threshold)
        output_masks.append(voted_mask)
        images.append(image)
    for i, mask in enumerate(output_masks):
        image = images[i]
        image = np.array(image.convert("RGBA"))
        mask = np.array(mask)
        #把mask给resize到和image一样的大小
        mask = cv2.resize(mask, (image.shape[1], image.shape[0])).astype(np.uint8)
        #mask大于0的地方，image的alpha通道为255，否则为0
        image[:,:,3] = mask*255
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        output_images.append(image)

    

    return output_masks, output_images

#主函数
def main_func(parse):
    save_path = parse.save_path

    if parse.sam_model == "vit_l":
        sam = sam_model_registry["vit_l"](checkpoint="./segany/sam_vit_l_0b3195.pth")
        print("Use vit_l")
    elif parse.sam_model == "vit_h":
        sam = sam_model_registry["vit_h"](checkpoint="./segany/sam_vit_h_4b8939.pth")
        print("Use vit_h")
    else:
        sam = sam_model_registry["vit_b"](checkpoint="./segany/sam_vit_b_01ec64.pth")
        print("Use vit_b")
    if parse.cuda:
        sam = sam.cuda()

    sam_model_generator = SamAutomaticMaskGenerator(sam,
                                                pred_iou_thresh=0.6,
                                                stability_score_thresh=0.8,
                                                crop_n_points_downscale_factor=1,
                                                crop_n_layers=1)
    yolo_model = YOLO(parse.yolo_model)

    output_masks, output_images = pipeline( sam_model_generator, yolo_model, parse.source, parse.threshold)
    if output_masks is None:
        print("No object detected!")
        return
    #保存图片
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_names = []
    # 如果source 是文件夹:
    if os.path.isdir(parse.source):
        for image_ in os.listdir(parse.source):
            image_names.append(image_)
    else:
        image_names.append(parse.source)
    for i, image in enumerate(output_images):
        print("Save {}".format(os.path.join(save_path, image_names[i].split("\\")[-1].split(".")[0]+'.png')))
        cv2.imwrite(os.path.join(save_path, image_names[i].split("\\")[-1].split(".")[0]+'.png'), image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='sample.jpg', help='path to the image')
    parser.add_argument('--threshold', type=float, default=0.3, help='threshold for mask')
    parser.add_argument('--save_path', type=str, default='./seg_output', help='path to save the mask')
    parser.add_argument('--sam_model', type=str, default='vit_b', help='sam model')
    parser.add_argument('--yolo_model', type=str, default='./yolo/ver3.pt', help='yolo model')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    main_func(parser.parse_args())


