# Anime-seg-yolov8(WIP)
A segmentation project based on aniseg, trained on yolov8-seg
使用Ani-Seg的训练集，通过组合前景、背景以产生训练集。使用ultralytics的yolov8-seg进行训练，通过结合yolov8-seg和SAM实现更加精准的表现
## Project works
1.读取https://github.com/SkyTNT/anime-segmentation 提供的训练集，并模仿、制造YOLOv8-seg的训练集

2.训练模型

3.实现yolov8 detect+segment anything细化边界的seg任务

4.将上面的过程给封装起来
## TODO:
1.增强数据集（AI生成），实现更好的数据集制作（缺少男性动漫角色、CHIBI角色、多角色数据集，没有实现数据增强手段）

2.训练不同大小的模型（n、s、m大小的）

3.优化SAM的超参数

## model

YOLOv8m-v0.1:https://huggingface.co/AkitoP/Anime-yolov8-seg
