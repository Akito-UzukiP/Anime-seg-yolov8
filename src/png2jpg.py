import os
import cv2
path = "./data/masks_2/"
save_path = "./data/masks_3/"
if not os.path.exists(save_path):
    os.mkdir(save_path)
pngs = os.listdir(path)
for i in pngs:
    img = cv2.imread(path+i)
    cv2.imwrite(save_path+i[:-4]+".jpg", img)