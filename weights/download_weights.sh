#!/bin/bash
# Download common models
#attempt_download('weights/yolov5s.pt');
#attempt_download('weights/yolov5m.pt');
#attempt_download('weights/yolov5l.pt');
#attempt_download('weights/yolov5x.pt')

#

python -c 
from utils.google_utils import *;
attempt_download('weights/yolov5m.pt');
