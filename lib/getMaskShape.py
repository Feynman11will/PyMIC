



import os
import sys
import xlrd
import json
import pydicom as dicom
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage import measure
import PIL
from DrawLabel import read_dicom, draw_anno_3d, lumTrans
import numpy as np
import math

import matplotlib.image as mp

def loadJson(path):
    f = open(path, encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    all = json.load(f)
    f.close()
    return all


def getDcmPathList(path,ParantPath):
    tasks = loadJson(path)
    print()
    lengthOfTasks = len(tasks)
    dicomFileList = []
    haveEdgeList = []
    for idx , task in enumerate(tasks) :
        tokenList= task['other_info']['dataPath'].split('/')[-3:]

        token = '/'.join(tokenList)
        fullPath = os.path.join(ParantPath,token)
        dicomFileList.append(fullPath)
        if task['nodes']!=[]:
            haveEdgeList.append({fullPath:task})
    return dicomFileList ,haveEdgeList

def PrintList(list):
    for idx,li in enumerate(list):
        print(f"{idx}:{li}")

def getFullpaths(paths):
    fullNameList = []
    for path in paths:
        masks = os.listdir(path)
        for mask in masks:
            fullName =  os.path.join(path,mask)
            fullNameList.append(fullName)
    return fullNameList
if __name__=='__main__':
    pathTask_10493_0 = '/data1/wanglonglong/DataSet/DeepwiseLabelLymphNodes/task_10493_0.json'
    parentPath = '/data1/wanglonglong/DataSet/DeepwiseLabelLymphNodes/label/mask'

    dicomFileList ,haveEdgeList  = getDcmPathList(pathTask_10493_0, parentPath)
    paths = []
    for haveEdge in haveEdgeList:
        paths.append(list(haveEdge.keys())[0])
    fullNameList = getFullpaths(paths)
    PrintList(fullNameList)
    print(f"len(fullNameList)------>{len(fullNameList)}")
    outputSizeList = []
    for imgNpyPath in fullNameList:
        imgNpy = np.load(imgNpyPath)
        labels, nums = measure.label(imgNpy, return_num=True)
        props = measure.regionprops(labels[:, :])

        bboxs = []
        centers = []
        for prop in props:
            bboxs.append(prop.bbox)
            center = []
            for c in prop.centroid:
                center.append(math.floor(c))
            centers.append(center)
        for bbox in bboxs:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            outputSizeList.append([imgNpyPath,w, h])

    PrintList(outputSizeList)
    npListName = '/data1/wanglonglong/01workspace/PyMIC/examples/lymph/shapeList'
    np.save(npListName,outputSizeList)