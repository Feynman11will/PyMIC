import os
import json
import pydicom as dicom
import numpy as np
import cv2
import SimpleITK as sitk


def draw_anno_2d(img_fn, anno_fn, title = '', img_dir = None):
    img, dcm_files = read_dicom(img_fn)
    anno = json.load(open(anno_fn, 'r'))
    print(anno.keys())

    img_arr = sitk.GetArrayFromImage(img)[0]
    mask_arr = np.zeros_like(img_arr)

    for i, node in enumerate(anno['nodes']):
        rois = node["rois"]
        for roi in rois:
            for j in range(len(roi)):
                roi[j][0] = int(round(roi[j][0]))
                roi[j][1] = int(round(roi[j][1]))
            roi = np.array(roi)
            cv2.drawContours(mask_arr, (roi,), -1, 1, thickness=-1)
    if img_dir & (mask_arr.sum() > 0):
        # show_img_mask_2d(img_arr, mask_arr, title, img_dir)
        pass
    return img_arr, mask_arr

def read_dicom(dicom_path, sort = True):
    if os.path.isfile(dicom_path):
        return sitk.ReadImage(dicom_path), dicom_path
    reader = sitk.ImageSeriesReader()
    dcm_files = list(reader.GetGDCMSeriesFileNames(dicom_path))
    if sort:
        dcm_files.sort(key = lambda x: dicom.read_file(x).InstanceNumber)

    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    return img, dcm_files

def draw_anno_3d(img_fn, anno_fn, title = '', img_dir = None):
    img, dcm_files = read_dicom(img_fn, sort = True)
    # anno = json.load(open(anno_fn, 'r'))
    anno = anno_fn
    img_arr = sitk.GetArrayFromImage(img)
    print(img_arr.shape)
    mask_arr = np.zeros_like(img_arr)

    for i, node in enumerate(anno['nodes']):
        rois = node["rois"]
        for roi in rois:
            slice_index = roi['slice_index'] -1
            points = roi['edge']
            for j in range(len(points)):
                points[j][0] = int(round(points[j][0]))
                points[j][1] = int(round(points[j][1]))
            points = np.array(points)
            cv2.drawContours(mask_arr[slice_index], (points,), -1, 1, thickness=-1)
    if img_dir:
        if mask_arr.sum() > 0:
            # show_img_mask(img_arr, mask_arr, title, img_dir)
            pass
    return img_arr, mask_arr

def lumTrans(img, low = -600, high = 600, scale = 255):
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*scale).astype('float32')
    return newimg