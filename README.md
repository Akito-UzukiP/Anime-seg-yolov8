# Anime-seg-yolov8 project for CS308 Computer Vision
A segmentation project based on aniseg, trained on yolov8-seg

使用Ani-Seg的训练集，通过组合前景、背景以产生训练集。使用ultralytics的yolov8-seg进行训练，通过结合yolov8-seg和SAM实现更加精准的表现
![Sample](output.png)
![Sample2](output2.png)
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

## Generate dataset
In order to generate the training dataset, you should use the provided script with the following command line arguments:

- fg: This argument specifies the path to the folder containing the foreground images. By default, it points to the ./datasets/fg directory.

- bg: This argument specifies the path to the folder containing the background images. By default, it points to the ./datasets/bg directory.

- out: This argument specifies the path to the output directory where the generated images will be saved. By default, it points to the ./datasets/out directory.

- mask: This argument specifies the path to the folder where the masks for the foreground objects will be saved. By default, it points to the ./datasets/masks directory.

- max_item: This argument specifies the maximum number of foreground items that can be placed on one background image. By default, it is set to 8.

- pic_num: This argument specifies the maximum number of pictures to be generated. By default, it is set to 1.

To run the script with these arguments, use the following command:
```bash
python generate.py --fg path_to_foreground --bg path_to_background --out path_to_output --mask path_to_mask --max_item number_of_items --pic_num number_of_pictures
```
## 检测并分割动漫角色

使用命令行检测和分割动漫角色，如下：

```bash
python detect.py --source sample.jpg --threshold 0.3 --save_path ./seg_output --sam_model vit_l --yolo_model ./yolo/ver3.pt --cuda --mask --sam --yolo
```
各参数具体含义如下：

- source：输入图片的路径，默认为'sample.jpg'
- threshold：生成分割遮罩的阈值，默认为0.3
- save_path：保存结果的路径，默认为'./seg_output'
- sam_model：SAM模型，有 'vit_l', 'vit_h' 和 'vit_b' 三种可选，默认为 'vit_l'
- yolo_model：YOLO模型路径，默认为'./yolo/ver3.pt'
- cuda：如果需要使用cuda加速，加上此选项(显存占用非常高，请注意) 若使用CPU计算，一张图大约需要2分钟
- mask: 是否保存分割遮罩图片，默认为False
- sam: 是否保存sam分割结果图片，默认为False
- yolo: 是否保存yolo分割结果图片，默认为False


## 模型下载

- YOLOv8-seg:https://huggingface.co/AkitoP/Anime-yolov8-seg
- Segment Anything: https://github.com/facebookresearch/segment-anything
## Reference
- Segment Anything: https://github.com/facebookresearch/segment-anything
- YOLOv8: https://github.com/ultralytics/ultralytics
- anime-segmentation: https://github.com/SkyTNT/anime-segmentation
- semantic segment anything: https://github.com/fudan-zvg/Semantic-Segment-Anything
