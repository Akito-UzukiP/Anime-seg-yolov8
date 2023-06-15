import os
import cv2
import PIL.Image as Image
import numpy as np
#读取mask 返回yolo格式的str
def mask2yolo(mask_img, show=False):
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
    # 只保留最大的轮廓
    max_size = max(size)
    max_index = size.index(max_size)
    poligon = poligons[max_index]
    # 画出轮廓
    zeros = np.zeros(img.shape, np.uint8)
    print(np.array(poligon).shape)
    if show:
        cv2.fillPoly(zeros, [np.array(poligon).reshape(-1, 2)], (255, 255, 255))
        cv2.imshow("mask",zeros)
    #保存为yolo的str
    yolo_str = "0 "
    for j in range(0,len(poligon),2):
        yolo_str += str(poligon[j]/img.shape[1])+" "+str(poligon[j+1]/img.shape[0])+" "
    return yolo_str

# 读取fg bg mask 返回叠加后的fg bg 和相应变换后的mask 
def make_pic(fg,bg,mask):
    # 保证fg bg mask的都是PIL的Image格式
    assert isinstance(fg,Image.Image)
    assert isinstance(bg,Image.Image)
    assert isinstance(mask,Image.Image)
    
    # 先不考虑旋转了，只考虑平移
    do_rotation = False
    do_translation = True
    do_scale = False
    
    # 将fg resize到可以接受的大小，然后把fg叠加到bg上，然后根据这一步的操作将mask也变换到与bg一样的空白图片上
    # 1. resize fg
    # 计算比例 使fg的长宽都小于bg
    # 如果本来就小于bg就不用resize了
    if fg.size[0] > bg.size[0] or fg.size[1] > bg.size[1]:
        scale = min(bg.size[0]/fg.size[0],bg.size[1]/fg.size[1])
        # resize
        fg = fg.resize((int(fg.size[0]*scale),int(fg.size[1]*scale)))

    # 2. 将fg叠加到bg上
    # 随机生成一个位置
    # 生成的位置是fg的左上角的位置 需要保证右下角的位置在bg内
    max_x = bg.size[0] - fg.size[0]
    max_y = bg.size[1] - fg.size[1]
    # 生成随机位置
    new_x = np.random.randint(0,max_x)
    new_y = np.random.randint(0,max_y)
    # 将fg叠加到bg上 注意透明图层
    bg.paste(fg, (new_x, new_y), fg)

    # 3. 将mask也变换到与bg一样的全0图片上（无透明图层），然后将mask直接叠加上去
    print(mask)
    mask = mask.convert("L")
    blank_mask = Image.new("L",bg.size,0)
    blank_mask.paste(mask, (new_x, new_y), mask)
    

    return bg, blank_mask