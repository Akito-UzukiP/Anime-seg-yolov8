from src.utils import *
import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
from PIL import ImageEnhance
from tqdm import tqdm
import argparse
def generate(argparse):
    fg_path = argparse.fg
    bg_path = argparse.bg
    out = argparse.out
    mask_path = argparse.mask
    if not os.path.exists(out):
        os.makedirs(out)


    max_target_num = argparse.max_item
    pic_num = argparse.pic_num
    fgs = os.listdir(fg_path)
    bgs = os.listdir(bg_path)
    masks = os.listdir(mask_path)
    for n in tqdm(range(pic_num), desc='Processing images'):
        target_num = np.random.randint(1,max_target_num)
        choice = np.random.choice(len(fgs),target_num)
        fg = []
        mask = []
        for i in choice:
            fg.append(fgs[i])
            mask.append(masks[i])
        bg_cur = bgs[np.random.randint(0,len(bgs))]
        fg_list = [os.path.join(fg_path,i) for i in fg]
        bg = os.path.join(bg_path,bg_cur)
        mask_list = [os.path.join(mask_path,i) for i in mask]
        mixed_pic, mixed_mask, mask_sizes = make_pic(fg_list, bg, mask_list)
        filename = ""
        for i in fg:
            filename += i[:-4]+"_"
        random_name = np.random.randint(0,100000)
        mixed_pic.save(os.path.join(out,filename+"_"+bg_cur[:-4]+str(random_name)+".jpg"))
        #mixed_mask.save(mainpath+"mixed_mask/"+filename+"_"+bg_cur[:-4]+".jpg")
        yolo = mask2yolo(mixed_mask, mask_sizes, show=False)
        with open(os.path.join(out,filename+"_"+bg_cur[:-4]+str(random_name)+".txt"),"w") as f:
            f.write(yolo[0])
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fg', type=str, default='./datasets/fg', help='foreground')
    parser.add_argument('--bg', type=str, default='./datasets/bg', help='background')
    parser.add_argument('--out', type=str, default='./datasets/out', help='output')
    parser.add_argument('--mask', type=str, default='./datasets/masks', help='mask')
    parser.add_argument('--max_item', type=int, default=8, help='max item')
    parser.add_argument('--pic_num', type=int, default=1, help='max num')
    args = parser.parse_args()
    generate(args)