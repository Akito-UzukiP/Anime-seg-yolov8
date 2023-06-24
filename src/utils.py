import os
import cv2
import PIL.Image as Image
from PIL import ImageEnhance
import numpy as np
#读取mask 返回yolo格式的str
def mask2yolo(mask_img, mask_sizes,show = False):
    #如果是PIL的Image格式就转换成np的array
    if isinstance(mask_img,Image.Image):
        mask_img = np.array(mask_img)
    poligons = []
    size = []
    img = mask_img
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for object in contours:
        coords = []
        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        poligons.append(coords)
        size.append(len(coords))
    # 保留所有面积大于mask_sizes的轮廓
    min_mask_sizes = min(mask_sizes)
    #print("min_mask_size:"+str(min_mask_sizes))
    temp = []
    for i in range(len(poligons)):
        area = cv2.contourArea(np.array(poligons[i]).reshape(-1, 2))
        if area*1.1 > min_mask_sizes:
            temp.append(poligons[i])
            #print(area)
    poligons = temp
    # 画出轮廓
    zeros = np.zeros(img.shape, np.uint8)
    #保存为yolo的str
    yolo_str = ""
    if show:
        for poligon in poligons:
            cv2.fillPoly(zeros, [np.array(poligon).reshape(-1, 2)], (255, 255, 255))
    for poligon in poligons:
        yolo_str += "0 "
        for j in range(0,len(poligon),2):
            yolo_str += str(poligon[j]/img.shape[1])+" "+str(poligon[j+1]/img.shape[0])+" "
        yolo_str += "\n"
    return yolo_str, zeros, poligons

# 读取fg bg mask 返回叠加后的fg bg 和相应变换后的mask 
# 修改成多张fg mask
def make_pic(fgs_,bg,masks_,do_rotation=True,do_scale=True,do_flip=True,do_color_transformation=True,do_noise_injection=True, do_random_crop = True):
    # 读入的是路径，
    assert len(fgs_) == len(masks_), "前景图像与掩膜数量必须相等"
    fgs = []
    masks = []
    mask_sizes = []
    bg = Image.open(bg)
    for fg_, mask_ in zip(fgs_, masks_):
        fg = Image.open(fg_)
        mask = Image.open(mask_)
        fgs.append(fg)
        masks.append(mask)
    target_cnt = 0
    # 让bg扩大最大的fg的尺寸,扩大的部分用黑色填充

    #第一步，进行各项处理
    next_fgs = []
    next_masks = []
    for fg, mask in zip(fgs, masks):
        # 将fg resize到可以接受的大小，然后把fg叠加到bg上，然后根据这一步的操作将mask也变换到与bg一样的空白图片上
        # 1. resize fg
        # 计算比例 使fg的长宽都小于bg
        # 如果本来就小于bg就不用resize了
        if do_random_crop and not np.random.randint(0,10):

            # 随机裁剪
            # 先计算裁剪的大小
            crop_rate_w = np.random.uniform(0.5,1)
            crop_rate_h = np.random.uniform(0.5,1)
            crop_w = int(fg.size[0]*crop_rate_w)
            crop_h = int(fg.size[1]*crop_rate_h)
            # 随机裁剪
            x = np.random.randint(0,fg.size[0]-crop_w)
            y = np.random.randint(0,fg.size[1]-crop_h)
            fg = fg.crop((x,y,x+crop_w,y+crop_h))
            mask = mask.crop((x,y,x+crop_w,y+crop_h))
        if do_rotation:
            angle = np.random.randint(0,360)
            fg = fg.rotate(angle,resample=Image.BICUBIC,expand=True)
            mask = mask.rotate(angle,resample=Image.BICUBIC,expand=True)
        if do_scale:
            scale = np.random.uniform(0.5,1.5)
            # resize
            fg = fg.resize((int(fg.size[0]*scale),int(fg.size[1]*scale)))
            # mask也resize
            mask = mask.resize((int(mask.size[0]*scale),int(mask.size[1]*scale)))
        if do_flip:
            if np.random.randint(0,2):
                fg = fg.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if do_color_transformation:
            #给颜色加上随机的噪声
            fg = ImageEnhance.Color(fg).enhance(np.random.uniform(0.5,1.5))
            fg = ImageEnhance.Brightness(fg).enhance(np.random.uniform(0.5,1.5))
            fg = ImageEnhance.Contrast(fg).enhance(np.random.uniform(0.5,1.5))
            fg = ImageEnhance.Sharpness(fg).enhance(np.random.uniform(0.5,1.5))
        next_fgs.append(fg)
        next_masks.append(mask)

    # 第二步，将fgs和masks叠加到bg上
    original_bg_size = bg.size
    fg_sizes = []
    for fg in next_fgs:
        fg_sizes.append(fg.size)
    max_fg_size_x = max([fg_size[0] for fg_size in fg_sizes])
    max_fg_size_y = max([fg_size[1] for fg_size in fg_sizes])
    old_bg = bg
    bg = Image.new("RGB",(max_fg_size_x+original_bg_size[0],max_fg_size_y+original_bg_size[1]),(0,0,0))
    bg.paste(old_bg,(max_fg_size_x,max_fg_size_y))
    blank_mask = Image.new("L",bg.size,0)
    fgs = next_fgs
    masks = next_masks

    for fg, mask in zip(fgs, masks):
        fg_size = fg.size
        bg_size = bg.size



        min_x = max_fg_size_x - int(fg_size[0]/2)
        min_y = max_fg_size_y - int(fg_size[1]/2)
        max_x = bg_size[0] - int(fg_size[0]*1.5)
        max_y = bg_size[1] - int(fg_size[1]*1.5)
        # 生成随机位置, 高斯分布
        new_x = int(np.random.normal((max_fg_size_x+bg_size[0])/2, (bg_size[0]) / 3))
        new_y = int(np.random.normal((max_fg_size_y+bg_size[1])/2, (bg_size[1]) / 3))
        # 把new_x,new_y限制在0到max_x,max_y之间
        new_x = max(min_x, min(new_x, max_x))
        new_y = max(min_y, min(new_y, max_y))
        # 将fg叠加到bg上 注意透明图层
        # 判断是否叠加的时候叠加到了之前的mask上
        bg.paste(fg, (new_x, new_y), fg)


        # 3. 将mask也变换到与bg一样的全0图片上（无透明图层），然后将mask直接叠加上去
        mask = mask.convert("L")
        temp = Image.new("L",bg.size,0)
        temp.paste(mask,(new_x,new_y),mask)
        #if not np.logical_and(np.array(blank_mask),np.array(temp)).any():
        #    target_cnt += 1
        #计算叠加、裁剪后的mask的面积
        temp = temp.crop((max_fg_size_x,max_fg_size_y,max_fg_size_x+original_bg_size[0],max_fg_size_y+original_bg_size[1]))
        area = np.array(temp).sum()/255
        if area > 500:
            mask_sizes.append(np.array(temp).sum()/255)
        #判断是否叠加的时候叠加到了之前的mask上

        blank_mask.paste(mask, (new_x, new_y), mask)
    # 将bg裁剪到原来的大小
    bg = bg.crop((max_fg_size_x,max_fg_size_y,max_fg_size_x+original_bg_size[0],max_fg_size_y+original_bg_size[1]))
    blank_mask = blank_mask.crop((max_fg_size_x,max_fg_size_y,max_fg_size_x+original_bg_size[0],max_fg_size_y+original_bg_size[1]))
        # 噪声注入
    if do_noise_injection:
        noise = np.random.normal(0, np.random.randint(0,32), (bg.size[1], bg.size[0], 3))
        bg = Image.fromarray((np.array(bg) + noise).clip(0,255).astype(np.uint8))
    return bg, blank_mask, mask_sizes







# 添加随机噪声
def random_noise(image):
    return image + np.random.normal(0, 0.1, image.shape)

# 对比度调整
def adjust_contrast(image):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(np.random.uniform(0.5, 1.5))

# 亮度调整
def adjust_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(np.random.uniform(0.5, 1.5))

# 饱和度调整
def adjust_saturation(image):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(np.random.uniform(0.5, 1.5))