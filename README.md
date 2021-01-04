## Training

```bash
$ python train.py --img 640 --batch 4 --epochs 10 --data ./data/custom/custom.yaml --cfg ./models/custom_5m.yaml --weights 'weights/yolov5m.pt' --multi-scale --hyp ./data/custom/hyp_finetune.yaml

## yolov5m.pt download
https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt