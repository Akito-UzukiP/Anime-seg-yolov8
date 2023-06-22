# Anime-seg-yolov8 project for CS308 Computer Vision
A segmentation project based on aniseg, trained on yolov8-seg
使用Ani-Seg的训练集，通过组合前景、背景以产生训练集。使用ultralytics的yolov8-seg进行训练，通过结合yolov8-seg和SAM实现更加精准的表现
## Project works
1.实现了一个简单的数据集生成器，可以通过组合前景、背景以产生训练集

2.在基于ani-seg生成的训练集和coco2017(仅person)上训练了yolov8m-seg模型

3.实现yolov8 detect+segment anything细化边界的seg任务

### What can this repo do?
- Generate dataset
- Use pretrained model to detect and segment anime character

## Requirements
- 安装python>=3.8以及requirements.txt中的依赖
- 将Segment Anything模型放入./segany 文件夹中
- 将YOLOv8-seg模型放入 ./yolo 文件夹中

## How to generate dataset
TODO

## 如何检测并分割动漫角色

使用命令行检测和分割动漫角色，如下：

```bash
python detect.py --source [图片路径] --threshold [分割阈值] --save_path [保存路径] --sam_model [SAM模型] --yolo_model [YOLO模型路径] [--cuda]
```
各参数具体含义如下：

- source：输入图片的路径，默认为'sample.jpg'
- threshold：生成分割遮罩的阈值，默认为0.3
- save_path：保存结果的路径，默认为'./seg_output'
- sam_model：SAM模型，有 'vit_l', 'vit_h' 和 'vit_b' 三种可选，默认为 'vit_l'
- yolo_model：YOLO模型路径，默认为'./yolo/ver3.pt'
- cuda：如果需要使用cuda加速，加上此选项

## TODO:

## 模型下载

YOLOv8-seg:https://huggingface.co/AkitoP/Anime-yolov8-seg
Segment Anything: https://github.com/facebookresearch/segment-anything
## Reference
- Segment Anything: https://github.com/facebookresearch/segment-anything
- YOLOv8: https://github.com/ultralytics/ultralytics
- anime-segmentation: https://github.com/SkyTNT/anime-segmentation
