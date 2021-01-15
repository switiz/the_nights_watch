#YOLO v5 Xray object detection (�ѱ� ������Ʈ)

- �������б� �ΰ����� ���п� AiHub dataset�� �̿��� Xray object detection ������Ʈ�� ���  

----

##���� 

- ���п����� ������Ʈ�� �����ϸ� YOLO �˰����� ����ġ�� �� ��� ���� �ڷ� �� ��ü�� ���� �ʰ� �ִ� �ϴ��� �ڷᰡ ������ ������ �ӹ��� �ֽ��ϴ�.    


- �ܱ��� �ڷ�� ���� ���� �������� �����ϴ� �ڷᰡ ���� �ʰ� ����ȭ�Ǿ��־� �ش� git repo�� ���� �ڷḦ �����Ͽ� �����ϴ� �Ϳ� 1�� ������ �ΰ��� �մϴ�.   


- �ڵ��� �ǹ̿� ��� ��ġ�� custom data�� Train�ϰ� ���� Ȱ���� �� �ִ��� Tutorial�� �����ϰ� Function�� �ѱ� �ּ��� �����ϴ°��� �� ������Ʈ�� ���� �����Դϴ�.   


----
##������
![img.png](img.png)

---
## �н�

~~~
python train.py --img 640 --batch 4 --epochs 30 --data ./data/custom/custom.yaml --cfg ./models/custom_5m.yaml --weights ./weights/yolov5m.pt
~~~
---

## ���
model : custom_5m  
epochs: 100 epoch    

| Class                         | mAP@.5 | mAP@.5:.95: |
|-------------------------------|--------|-------------|
| all                           | 0.988  | 0.883       |

----

## ����
yolov3(pytorch): https://github.com/ultralytics/yolov3


yolov4(c++): https://github.com/AlexeyAB/darknet


yolov5(pytorch): https://github.com/ultralytics/yolov5  



## Pre-Trained Data 
yolov5m.pt download  
$ python train.py --img 640 --batch 4 --epochs 30 --data ./data/custom/custom.yaml --cfg ./models/custom_5m_p2.yaml --weights 'weights/yolov5m.pt'
'''

## yolov5m.pt download
https://github.com/ultralytics/yolov5/releases/download/v3.0/yolov5m.pt

