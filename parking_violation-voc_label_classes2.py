import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#sets=[('2018', 'train'), ('2018', 'val'), ('2018', 'test')]
sets=[('2021', 'train')]

#classes = ["bus","sedan","truck","van"]
#classes = ["lpr","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
#classes = ["lpr","c0","f0","s0",".","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
#classes = ["screw","corner",".","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
#classes = ["Leave_Car","Enter_Car","Plate","Car"]
classes = ["Parking_Violation", "lpr"]

def convert(size, box):
    print(image_id)
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('parking_violation/datasets/label/%s.xml'%(image_id))
    out_file = open('parking_violation/datasets/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('parking_violation/datasets/labels/'):
        os.makedirs('parking_violation/datasets/labels/')
    image_ids = open('parking_violation/datasets/2021train.txt').read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/parking_violation/datasets/image/%s.jpg\n'%(wd,image_id))
        convert_annotation(year, image_id)
    list_file.close()

