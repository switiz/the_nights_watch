## Training

```bash
$ python train.py --img 640 --batch 4 --epochs 10 --data ./data/custom/custom.yaml --cfg ./models/custom_5m.yaml --weights 'weights/yolov5m.pt' --multi-scale --hyp ./data/custom/hyp_finetune.yaml
