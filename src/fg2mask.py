import os
import cv2
path = "./datasets/new/"
mask_path = "./datasets/new_masks/"

if not os.path.exists(mask_path):
    os.mkdir(mask_path)
cnt = 0
for i in os.listdir(path):
    #带有alpha通道的图片
    img = cv2.imread(path+i, cv2.IMREAD_UNCHANGED)
   # print(img)
    # 将图片转化为纯黑白的，非0即255
    #print(img[:,:,3])
    ret, thresh = cv2.threshold(img[:,:,3], 0, 255, cv2.THRESH_BINARY)
    #print(thresh.shape)
    cv2.imwrite(mask_path+i, thresh)
    cnt += 1
