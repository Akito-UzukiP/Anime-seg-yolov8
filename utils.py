import os
import cv2
import PIL.Image as Image
from PIL import ImageEnhance
import numpy as np
#读取mask 返回yolo格式的str
def mask2yolo(mask_img, item_nums,show = False):
    #如果是PIL的Image格式就转换成np的array
    if isinstance(mask_img,Image.Image):
        mask_img = np.array(mask_img)
    poligons = []
    size = []
    img = mask_img
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for object in contours:
        coords = []
        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))
        poligons.append(coords)
        size.append(len(coords))
    # 保留最大的item_nums个轮廓
    #print(np.argsort(size)[-item_nums:])
    print(np.argsort(size)[-item_nums:])
    temp = []
    for i in np.argsort(size)[-item_nums:]:
        temp.append(poligons[i])
    poligons = temp
    # 画出轮廓
    zeros = np.zeros(img.shape, np.uint8)
    print(len(poligons))
    #保存为yolo的str
    yolo_str = ""
    if show:
        for poligon in poligons:
            cv2.fillPoly(zeros, [np.array(poligon).reshape(-1, 2)], (255, 255, 255))
        cv2.waitKey(0)
    for poligon in poligons:
        yolo_str += "0 "
        for j in range(0,len(poligon),2):
            yolo_str += str(poligon[j]/img.shape[1])+" "+str(poligon[j+1]/img.shape[0])+" "
        yolo_str += "\n"
    return yolo_str, zeros

# 读取fg bg mask 返回叠加后的fg bg 和相应变换后的mask 
# 修改成多张fg mask
def make_pic(fgs_,bg,masks_,do_rotation=True,do_scale=True,do_flip=True,do_color_transformation=True,do_noise_injection=True):
    # 读入的是路径，
    assert len(fgs_) == len(masks_), "前景图像与掩膜数量必须相等"
    fgs = []
    masks = []
    bg = Image.open(bg)
    for fg_, mask_ in zip(fgs_, masks_):
        fg = Image.open(fg_)
        mask = Image.open(mask_)
        fgs.append(fg)
        masks.append(mask)
    # 先不考虑旋转了，只考虑平移
    blank_mask = Image.new("L",bg.size,0)
    target_cnt = 0
    for fg, mask in zip(fgs, masks):
        # 将fg resize到可以接受的大小，然后把fg叠加到bg上，然后根据这一步的操作将mask也变换到与bg一样的空白图片上
        # 1. resize fg
        # 计算比例 使fg的长宽都小于bg
        # 如果本来就小于bg就不用resize了
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
            enhancer = ImageEnhance.Color(fg)
            fg = enhancer.enhance(np.random.uniform(0.5, 1.5))  # 对颜色进行增强，值大于1表示增强，小于1表示减弱



        if fg.size[0] >= bg.size[0] or fg.size[1] >= bg.size[1]:
            scale = min(bg.size[0]/fg.size[0],bg.size[1]/fg.size[1])/1.2
            # resize
            fg = fg.resize((int(fg.size[0]*scale),int(fg.size[1]*scale)))
            # mask也resize
            mask = mask.resize((int(mask.size[0]*scale),int(mask.size[1]*scale)))
        # 防止fg太小 至少得是bg的一半大小
        if fg.size[0] < bg.size[0]/2 and fg.size[1] < bg.size[1]/2:
            scale = min(bg.size[0]/fg.size[0],bg.size[1]/fg.size[1])/1.5
            # resize
            fg = fg.resize((int(fg.size[0]*scale),int(fg.size[1]*scale)))
            # mask也resize
            mask = mask.resize((int(mask.size[0]*scale),int(mask.size[1]*scale)))

        # 随机生成一个位置
        # 生成的位置是fg的左上角的位置 需要保证右下角的位置在bg内
        max_x = bg.size[0] - fg.size[0]
        max_y = bg.size[1] - fg.size[1]
        #print(max_x,max_y)
        # 生成随机位置
        new_x = np.random.randint(0,max_x)
        new_y = np.random.randint(0,max_y)
        # 将fg叠加到bg上 注意透明图层
        # 判断是否叠加的时候叠加到了之前的mask上
        bg.paste(fg, (new_x, new_y), fg)

        # 3. 将mask也变换到与bg一样的全0图片上（无透明图层），然后将mask直接叠加上去
        #print(mask)
        mask = mask.convert("L")
        temp = Image.new("L",bg.size,0)
        temp.paste(mask,(new_x,new_y),mask)
        if not np.logical_and(np.array(blank_mask),np.array(temp)).any():
            target_cnt += 1
        #判断是否叠加的时候叠加到了之前的mask上

        blank_mask.paste(mask, (new_x, new_y), mask)
        
        # 噪声注入
        if do_noise_injection:
            noise = np.random.normal(0, np.random.randint(0,32), (bg.size[1], bg.size[0], 3))
            bg = Image.fromarray((np.array(bg) + noise).clip(0,255).astype(np.uint8))
    return bg, blank_mask, target_cnt


