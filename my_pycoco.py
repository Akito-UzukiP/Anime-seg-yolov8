import os
import cv2
import numpy as np
mask_path = "./data/mask_test/train_mask/"
for i in os.listdir(mask_path):
    poligons = []
    size = []
    img = cv2.imread(mask_path+i,cv2.IMREAD_GRAYSCALE)
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
    #保存为yolo格式
    with open("./data/mask_test/train_mask/"+i[:-4]+".txt","w") as f:
        for j in range(0,len(poligon),2):
            f.write("0 ")
            f.write(str(poligon[j]/img.shape[1])+" "+str(poligon[j+1]/img.shape[0]))
    #cv2.fillConvexPoly(zeros, np.array(poligon).reshape(-1, 2), 255)

    #cv2.imshow("a", img)
    #cv2.waitKey(1000)