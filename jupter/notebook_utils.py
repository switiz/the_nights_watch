import json
import os
from collections import defaultdict
from pycocotools.coco import COCO
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, ImageFont
import glob as glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils

def annToRLE(ann):
    """
    copy from coco api for fixed exception case
    """
    h, w = 1050, 1680
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        try:
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        except:
            print('segm:', segm, ann['image_id'])
            return None
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def segtoBox (seg):
    return maskUtils.toBbox(seg)

def check_dir(path):
    """
    path select
    """
    if path == 'colab':
        images_dir_path = '/content/drive/' + 'Shared drives' + '/YS_NW/2.Data/Train/Data'
        json_file_path = '/content/drive/Share|d drives/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'google_drive':
        images_dir_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Data'
        json_file_path = 'G:/공유 드라이브/YS_NW/2.Data/Train/Meta/CoCo/coco_rapiscan.json'
    elif path == 'local_d':
        images_dir_path = 'D:/Local/Train/Data'
        json_file_path = 'E:/Project/coco_rapiscan_change.json'
    elif path == 'local_c':
        images_dir_path = 'C:/Local/Train/Data'
        json_file_path = 'E:/Project/coco_rapiscan_change.json'
    elif path == 'new_d':
        images_dir_path = 'D:/Local/New'
        json_file_path = 'E:/Project/coco_rapiscan_change.json'
    elif path =='new_c':
        images_dir_path = 'C:/Local/Train'
        json_file_path = 'E:/Project/coco_rapiscan_change.json'
    elif path == 'eval_e':
        images_dir_path = 'E:/Project/Eval'
        json_file_path = 'E:/Project/coco_eval_rapiscan_change.json'
    elif path == 'local_e':
        images_dir_path = 'E:/Project/Train'
        json_file_path = 'E:/Project/coco_rapiscan_change.json'

    return images_dir_path, json_file_path

def select_class(path, classes):
    """
    class selection
    """
    for x in classes:
        if x in path:
            return True
        else:
            continue
    return False

def normalize (input, size):
    ratio_height, ratio_width = float(size[0]/1050), float(size[1]/1680)
    input[0], input[1], input[2], input[3] = input[0]*ratio_width, input[1]*ratio_height, input[2]*ratio_width, input[3]*ratio_height
    return input

def convert_labels(input, size=None):
    """
    COCO to YOLO Labels
    """
    if size is None:
        size = [1050, 1680]
        x1, y1, x2, y2 = input[0], input[1], input[2], input[3]
    else :
        input = normalize(input, size)
        x1, y1, x2, y2 = input[0], input[1], input[2], input[3]
    xmin = x1
    xmax = x1 + x2
    ymin = y1
    ymax = y1 + y2
    dh = 1. / size[0]
    dw = 1. / size[1]
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    #print('input : x1:',x1,',y1:',y1,',x2:',x2,',y2:',y2)
    #print('output : x:',x,',y:',y,',w:',w,',h:',h)
    return (x, y, w, h)

def index_change(index):
    """
    Index change
    Xray index start 0 and index 14, 35 is empty
    """
    if index <=13:
        index-=1
    elif index >14 and index <=34:
        index-=2
    elif index >35:
        index-=3
    return int(index)

"""
Not used
Yolo to original
def from_yolo_to_cor(box):
    height, width = 1050, 1680
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] + box[2] / 2) * width), int((box[1] + box[3] / 2) * height)
    x2, y2 = int((box[0] - box[2] / 2) * width), int((box[1] - box[3] / 2) * height)
    return x1, y1, x2, y2
"""

def from_yolo_to_cor(box):
    """
    to check yolo value
    """
    height, width = 1050, 1680
    x,y,w,h = box[0], box[1], box[2], box[3]
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x = x*width
    w = w*width
    y = y*height
    h = h*height
    x1 = int(x-w/2)
    x2 = int(x+w/2 -x1)
    y1 = int(y-h/2)
    y2 = int(y+h/2 - y1)
    return x1, y1, x2, y2

def make_coco_to_yolo(cat_types, images_dir_path, coco, annotations, train, size=None):
    name_box_id = defaultdict(list)
    for ano in tqdm(annotations):
        id = ano['image_id']
        # name = os.path.join(images_dir_path, images[id]['file_name'])
        ann_ids = coco.getAnnIds(imgIds=id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(coco_annotation[0]["image_id"])
        #cat = coco.getCatIds(catIds=id)
        if train:
            file_path = img_info[0]["path"].split('\\', maxsplit=7)[-1]
        else:
            file_path = img_info[0]["path"].split('\\', maxsplit=8)[-1]
        if select_class(file_path, cat_types) is False:
            continue
        if os.path.isfile(os.path.join(images_dir_path, file_path)) is False:
            #log(Tag , 'empty file : '+str(file_path))
            continue
        name = os.path.join(images_dir_path, file_path).replace('\\', '/')
        name_box_id[name].append([convert_labels(ano['bbox'], size), index_change(ano['category_id'])])
    return name_box_id

def write_anno_file(name_box_id, output, individual):
    """write to txt"""
    if individual:
        for key in tqdm(name_box_id.keys()):
            path = key[:-4]+'.txt'
            with open(path, 'w', encoding='utf-8') as f:
                box_infos = name_box_id[key]
                for idx, info in enumerate(box_infos):
                    x = info[0][0]
                    y = info[0][1]
                    w = info[0][2]
                    h = info[0][3]
                    c = info[1]
                    box_info = "%d %f %f %f %f" % (
                        c, x, y, w, h)
                    f.write(box_info)
                    if idx!=len(box_infos)-1:
                        f.write('\n')
            f.close()
    else:
        with open(output, 'w', encoding='utf-8') as f:
            for key in tqdm(name_box_id.keys()):
                f.write(key)
                box_infos = name_box_id[key]
                for idx, info in enumerate(box_infos):
                    x = info[0][0]
                    y = info[0][1]
                    w = info[0][2]
                    h = info[0][3]
                    c = info[1]
                    box_info = " %f,%f,%f,%f,%d" % (
                        x, y, w, h, c)
                    f.write(box_info)
                    if idx!=len(box_infos)-1:
                        f.write('\n')
        f.close()


def copy_file(name_box_id, images_dir_path, out_path):
    print('copy_file')
    for key in tqdm(name_box_id.keys()):
        path = key.replace(images_dir_path, out_path)
        dirname = os.path.dirname(os.path.abspath(path))
        if os.path.exists(path):
            continue
        #print(dirname)
        if os.path.isdir(dirname) is False:
            os.makedirs(dirname)
        copyfile(key, path)


def write_anno_file(name_box_id):
    """write to txt"""
    for key in tqdm(name_box_id.keys()):
        path = key[:-4]+'.txt'
        with open(path, 'w', encoding='utf-8') as f:
            box_infos = name_box_id[key]
            for idx, info in enumerate(box_infos):
                x = info[0][0]
                y = info[0][1]
                w = info[0][2]
                h = info[0][3]
                c = info[1]
                box_info = "%d %f %f %f %f" % (
                    c, x, y, w, h)
                f.write(box_info)
                if idx!=len(box_infos)-1:
                    f.write('\n')
        f.close()

def write_path_file(train, val, output, location):
    print('write_path_file')
    v3_path = location
    os.makedirs(v3_path, exist_ok=True)
    train_output = os.path.join(v3_path, 'train_'+output)
    val_output = os.path.join(v3_path, 'val_'+output)
    print(train_output)
    """write to txt"""
    with open(train_output, 'w', encoding='utf-8') as f:
        keys = train.keys()
        for idx, key in enumerate(keys):
            f.write(key)
            if idx !=len(keys) -1:
                f.write('\n')
    f.close()
    with open(val_output, 'w', encoding='utf-8') as f:
        keys = val.keys()
        for idx, key in enumerate(keys):
            f.write(key)
            if idx !=len(keys) -1:
                f.write('\n')
    f.close()
    print('write done')

"""
split tran / val
split by groups (ex: Battery_Single Others)
"""
def train_val_split(name_box_id):
    bbox =[]
    for key in name_box_id.keys():
        bbox.append(name_box_id[key])
    x = list(name_box_id.keys())

    x_train, x_val, y_train, y_val = train_test_split(x, bbox, train_size=0.95, random_state=True)

    train_data = defaultdict(list)
    for i in range(len(x_train)):
        train_data[x_train[i]].append(y_train[i])

    val_data = defaultdict(list)
    for i in range(len(x_val)):
        val_data[x_val[i]].append(y_val[i])

    print('split train : ', len(train_data), ' val:',len(val_data))
    return train_data, val_data

def make_label(cat_types, images_dir_path, coco, annotations):
    name_box_id = defaultdict(list)
    for ant in tqdm(annotations):
        id = ant['image_id']
        #name = os.path.join(images_dir_path, coco.loadImgs(id)[0]['file_name'])
        ann_ids = coco.getAnnIds(imgIds=id)
        coco_annotation = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(coco_annotation[0]["image_id"])
        #cat = coco.getCatIds(catIds=id)
        file_path = img_info[0]["path"].split('\\', maxsplit=7)[-1]
        #print(file_path)
        if select_class(file_path, cat_types) is False:
            continue
        if os.path.isfile(os.path.join(images_dir_path, file_path)) is False:
            # log(Tag , 'empty file : '+str(file_path))
            continue
        name = os.path.join(images_dir_path, file_path).replace('\\', '/')
        name_box_id[name].append([(ant['bbox']), ant['category_id']])
        #print(name)
    #print(name_box_id)
    return name_box_id


def class_id_to_str(class_id):
    class_dict = {34: 'ZippoOil', 37: 'Chisel', 24: 'Scissors', 30: 'SupplymentaryBattery', 22: 'PrtableGas',
                  36: 'Plier', 15: 'Knife', 17: 'Lighter', 11: 'Hammer', 9: 'Gun', 20: 'MetalPipe', 25: 'Screwdriver',
                  4: 'Axe', 28: 'Spanner', 23: 'Saw', 10: 'GunParts', 1: 'Aerosol', 19: 'Match', 2: 'Alcohol',
                  39: 'Electronic cigarettes(Liquid)', 12: 'HandCuffs', 41: 'Throwing Knife', 32: 'Thinner',
                  40: 'stun gun', 38: 'Electronic cigarettes', 26: 'SmartPhone', 13: 'HDD', 27: 'SolidFuel',
                  6: 'Battery', 3: 'Awl', 18: 'Liquid', 33: 'USB', 31: 'TabletPC', 29: 'SSD', 21: 'NailClippers',
                  16: 'Laptop', 7: 'Bullet', 8: 'Firecracker', 5: 'Bat'}

    return class_dict[class_id]

def label_print_to_img(name_box_id):
    fontsize = 14
    # outpath = './out/train_imgcheck/'
    # curpath = 'D:/Local/Train/Data/'

    outpath = 'C:/train_imgcheck/'
    curpath = 'C:/Local/Train/Data'

    font = ImageFont.truetype("arial.ttf", fontsize)
    for idx, key in enumerate(tqdm(name_box_id.keys())):
        img = Image.open(key)
        infos = name_box_id[key]
        draw = ImageDraw.Draw(img)
        for info in infos:
            data = info[0]
            class_name = class_id_to_str(info[1])
            xmin, ymin, xmax, ymax = data[0], data[1], data[0] + data[2], data[1] + data[3]
            rect = [xmin, ymin, xmax, ymax]
            label_rect = [xmin + 2, ymin + 1]
            draw.rectangle(rect, outline='red', width=3)
            draw.text(label_rect, class_name, fill='Blue', font=font)
        key = key.replace(curpath, outpath)
        path = os.path.split(key)
        if os.path.exists(path[0]) is False:
            os.makedirs(path[0])
        width, height = img.size
        ratio = float(height / width)
        newheight = int(ratio * 800)
        img = img.resize((800, newheight), Image.ANTIALIAS)
        img.save(key)

def write_eval_path_file(evaluation, output, location):
    print('write_eval_path_file')
    v3_path = location
    os.makedirs(v3_path, exist_ok=True)
    eval_output = os.path.join(v3_path, 'evaluation_'+output)
    """write to txt"""
    with open(eval_output, 'w', encoding='utf-8') as f:
        keys = evaluation.keys()
        for idx, key in enumerate(keys):
            f.write(key)
            if idx !=len(keys) -1:
                f.write('\n')
    f.close()
    print('write done')

def resize_file(path, img_size):
    filelist = glob.glob(path)
    filelist = [x for x in filelist if x.endswith('png')]
    print('resize total file length:', len(filelist))
    for file in tqdm(filelist):
        try:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            if img.shape[0] is not img_size:
                img = cv2.resize(img, dsize=(img_size,img_size), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(file, img)
        except:
           print(file)

    print('resize done')
