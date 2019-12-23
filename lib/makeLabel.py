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
def loadXls(path):

    xls_file = xlrd.open_workbook(path)

    return xls_file
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

def readDicom(dicomPath):
    img, dcm_files = read_dicom(dicomPath)


def showLabel(parentPath,pathOfTask,outputParentPath, showImg= False,saveImg= False):
    _, haveEdgeDicomList = getDcmPathList(pathOfTask, parentPath)
    print(len(haveEdgeDicomList))
    labelPath = os.path.join(outputParentPath, 'label')
    imgPath = os.path.join(labelPath, 'image')
    maskPath = os.path.join(labelPath, 'mask')
    if not os.path.exists(labelPath):
        os.mkdir(labelPath)
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        if not os.path.exists(maskPath):
            os.mkdir(maskPath)
    outputSizeList = []
    for idx, haveEdgeDicom in enumerate(haveEdgeDicomList):
        # if idx ==0:
        img_fn = list(haveEdgeDicom.keys())
        # print(img_fn[0])
        annos = list(haveEdgeDicom.values())
        annos=  annos[0]
        img_arr, mask_arr = draw_anno_3d(img_fn[0], annos)
        img_arr = lumTrans(img_arr, -250, 250)
        print(img_arr.shape)
        patientID = annos['patientID']
        studyUID = annos['studyUID']
        seriesUID  = annos[ "seriesUID"]
        casePath = os.path.join(patientID,studyUID,seriesUID)
        fullImgPath = os.path.join(imgPath, casePath)
        fullmaskPath = os.path.join(maskPath, casePath)

        if not os.path.exists(fullImgPath):
            os.makedirs(fullImgPath)
        if not os.path.exists(fullmaskPath):
            os.makedirs(fullmaskPath)

        imglist = []

        for i in range(len(img_arr)):
            if mask_arr[i].max() > 0:
                imglist.append(i)
                if showImg:
                    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                    ax[0].imshow(img_arr[i], 'gray')

                    contours = measure.find_contours(mask_arr[i], 0)
                    for n, contour in enumerate(contours):
                        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=1)

                    ax[1].imshow(mask_arr[i])
                    plt.show()
                    plt.close()


        if saveImg:
            for index,imgidx in enumerate(imglist):
                img = img_arr[imgidx]
                mask = mask_arr[imgidx]
                # labels, nums = measure.label(mask, return_num=True)

                fnameImg = os.path.join(fullImgPath,f"image_{str(index)}")
                fnamemask = os.path.join(fullmaskPath, f"mask_{str(index)}")
                print(fnameImg)
                print(fnamemask)
                print('-'*50)
                np.save(fnameImg,img)
                np.save(fnamemask,mask)


                labels, nums = measure.label(mask, return_num=True)
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
                    w = bbox[2]- bbox[0]
                    h = bbox[3] - bbox[1]
                    outputSizeList.append([w,h])
    outputSizeList = np.array(outputSizeList)
    fShapename = '/data1/wanglonglong/01workspace/PyMIC/examples/lymph/shapeList'
    np.save(fShapename, outputSizeList)
    return imglist,mask_arr,img_arr

def getStatics(parentPath,pathOfTask,outputParentPath,StaticList):
    _, haveEdgeDicomList = getDcmPathList(pathOfTask, parentPath)
    haveEdgeDicomList  = [haveEdgeDicom for haveEdgeDicom  in haveEdgeDicomList if (list(haveEdgeDicom.values())[0]['patientID'] in StaticList)]
    print(len(haveEdgeDicomList))
    labelPath = os.path.join(outputParentPath, 'label')
    imgPath = os.path.join(labelPath, 'image')
    maskPath = os.path.join(labelPath, 'mask')
    if not os.path.exists(labelPath):
        os.mkdir(labelPath)
        if not os.path.exists(imgPath):
            os.mkdir(imgPath)
        if not os.path.exists(maskPath):
            os.mkdir(maskPath)

    OutPutStaticsList = []
    for idx, haveEdgeDicom in enumerate(haveEdgeDicomList):
        # if idx ==0:
        img_fn = list(haveEdgeDicom.keys())
        # print(img_fn[0])
        annos = list(haveEdgeDicom.values())
        annos=  annos[0]
        img_arr, mask_arr = draw_anno_3d(img_fn[0], annos)
        img_arr = lumTrans(img_arr, -250, 250)
        print(img_arr.shape)
        patientID = annos['patientID']
        studyUID = annos['studyUID']
        seriesUID  = annos[ "seriesUID"]
        casePath = os.path.join(patientID,studyUID,seriesUID)
        fullImgPath = os.path.join(imgPath, casePath)
        fullmaskPath = os.path.join(maskPath, casePath)

        if not os.path.exists(fullImgPath):
            os.makedirs(fullImgPath)
        if not os.path.exists(fullmaskPath):
            os.makedirs(fullmaskPath)

        imglist = []

        for i in range(len(img_arr)):
            if mask_arr[i].max() > 0:
                imglist.append(i)



        for index,imgidx in enumerate(imglist):
            img = img_arr[imgidx]
            mask = mask_arr[imgidx]
            # labels, nums = measure.label(mask, return_num=True)

            fnameImg = os.path.join(fullImgPath,f"image_{str(imgidx)}")
            fnamemask = os.path.join(fullmaskPath, f"mask_{str(imgidx)}")
            print(fnameImg)
            print(fnamemask)
            print('-'*50)
            OutPutStaticsList.append([fnamemask])
            labels, nums = measure.label(mask, return_num=True)
            props = measure.regionprops(labels[:, :])

            bboxs = []
            centers = []

            for prop in props:
                bboxs.append(prop.bbox)
                center = []
                for c in prop.centroid:
                    center.append(math.floor(c))
                centers.append(center)

            outputSizeList = []
            for bbox in bboxs:
                w = bbox[2]- bbox[0]
                h = bbox[3] - bbox[1]
                outputSizeList.append([w,h])

            OutPutStaticsList[-1].append([centers,bboxs])
        # OutPutStaticsList = np.array(OutPutStaticsList)
    fShapename = '/data1/wanglonglong/01workspace/PyMIC/examples/lymph/staTcisList.txt'

    with open(fShapename, 'w+') as erg:
        for line in OutPutStaticsList:
            erg.writelines(str(line) + '\n')


if __name__=='__main__':

    pathTask_10493_0='/data1/wanglonglong/DataSet/DeepwiseLabelLymphNodes/task_10493_0.json'
    parentPath ='/data1/wanglonglong/DataSet/DeepwiseLabelLymphNodes/auto_rebuild'

    outputParentPath = '/data1/wanglonglong/DataSet/DeepwiseLabelLymphNodes'
    imglist,mask_arr,img_arr = showLabel(parentPath, pathTask_10493_0, outputParentPath, showImg=False, saveImg=True)
    getStatics(parentPath, pathTask_10493_0, outputParentPath)