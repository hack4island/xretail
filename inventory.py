from itertools import tee
import PIL
import torch.nn as nn
from torchvision.models import resnet18
import time
import traceback
import glob
from PIL import Image, ImageDraw, ImageOps

import torch
import numpy as np
from gtin import GTIN

import re
import cv2
import imutils
import argparse
from operator import itemgetter
import cv2
import pytesseract

import matplotlib.pyplot as plt

import random
import string

from ensemble_boxes import *
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import torchvision
import random
import pytesseract
from pytesseract import Output
import torch
from torchvision.ops import nms
import itertools
from itertools import chain
import cv2
from sklearn.cluster import KMeans



def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


model_pad = torch.hub.load('ultralytics/yolov5', 'custom', path='pad.pt')
model_ean = torch.hub.load('ultralytics/yolov5', 'custom', path='ean.pt')
#model_digit_box = torch.hub.load('ultralytics/yolov5', 'custom', path='digit_box_ean.pt')
#model_digit_recognition = tf.keras.models.load_model('digital_ia_3.h5')


def bloc_retrieval_pytesseract():
            im1_cv = np.array(im1)

            im1_cv_rgb = cv2.cvtColor(im1_cv, cv2.COLOR_BGR2RGB)
            results = pytesseract.image_to_data(im1_cv_rgb , output_type=Output.DICT)
            # loop over each of the individual text localizations
            for i in range(0, len(results["text"])):
            	# extract the bounding box coordinates of the text region from
            	# the current result
            	x = results["left"][i]
            	y = results["top"][i]
            	w = results["width"][i]
            	h = results["height"][i]
            	# extract the OCR text itself along with the confidence of the
            	# text localization
            	text = results["text"][i]
            	conf = int(results["conf"][i])

            	if conf > 0.4:
            		# display the confidence and text to our terminal
            		print("Confidence: {}".format(conf))
            		print("Text: {}".format(text))
            		print("")
            		# strip out non-ASCII text so we can draw the text on the image
            		# using OpenCV, then draw a bounding box around the text along
            		# with the text itself
            		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            		cv2.rectangle(im1_cv_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            		cv2.putText(im1_cv_rgb, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            			1.2, (0, 0, 255), 3)
            # show the output image
            #cv2.imwrite("{}_test_pytesseract_loc.jpeg".format(str(time.time())), im1_cv_rgb)



def encode_single_sample(img, label):
    img_width = 200
    img_height = 50
    img = tf.image.rgb_to_grayscale(img, name=None)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return {"image": img}


def enlarge(dim, percent):
    filt_dim = dim[:-2]
    w = filt_dim[2] - filt_dim[0]
    y = filt_dim[3] - filt_dim[1]
    filt_dim[0] = filt_dim[0] - w*percent
    filt_dim[1] = filt_dim[1] - y*percent*2
    filt_dim[2] = filt_dim[2] + w*percent
    filt_dim[3] = filt_dim[3] + y*percent*2
    return tuple(filt_dim)


def enlarge_s(dim, percent):
    filt_dim = dim
    w = filt_dim[2] - filt_dim[0]
    y = filt_dim[3] - filt_dim[1]
    filt_dim[0] = filt_dim[0] - w*percent
    filt_dim[1] = filt_dim[1] - y*percent
    filt_dim[2] = filt_dim[2] + w*percent
    filt_dim[3] = filt_dim[3] + y*percent
    return tuple(filt_dim)


custom_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
from pytesseract import Output



def box_list(box_lst):
    "add center for each box"
    pass



def get_center(box):
    x = box[0] + (box[2] - box[0])/2
    y = box[1] + (box[3] - box[1])/2
    return [x,y]


import numpy as np
import time
import re

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 2
# Red color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 3


def analyze(image_retail):

    fig = plt.figure()
    ax = plt.axes()

    img_ray = Image.open(image_retail)

    """extract pad"""
    output = model_pad(img_ray)

    im_fin = np.array(img_ray)

    """loop on price pads"""
    for pred in output.pred[0]:
        box_lst = []
        center_lst = []
        digit_lst = []

        dim = pred.tolist()
        im1 = img_ray.crop(enlarge(dim, 0.2))


        """extract EAN field"""
        output_ean = model_ean(im1)

        draw1 = ImageDraw.Draw(im1)

        for pre in output_ean.pred[0]:
            dim1 = pre.tolist()

            if dim1[-1] == 8.0:
                dim1[-1] = 7.0

            elif dim1[-1] == 9.0:
                dim1[-1] = 8.0

            elif dim1[-1] == 10:
                    dim1[-1] = 9.0


            if dim1[-2]>0.7:

                """select ration and pad"""
                if (float(dim1[0]-dim1[2] / dim1[1]/dim1[3]) > 10) and (dim1[-1] == 11.0):

                     im2 = im1.crop(enlarge(dim1, 0.1))
                     draw1.rectangle(enlarge(dim1, 0.10), outline=(random.randint(0, 255), 0, 0), width=2)
                     im2_arr = np.array(im2)

                     """extract digit boxes"""
                     img_rgb = cv2.cvtColor(im2_arr, cv2.COLOR_BGR2RGB)
                     im2_arr = img_rgb
                     res = pytesseract.image_to_string(im2_arr, config=custom_config)
                     #cv2.imwrite("./results/"+res+"_ocr_pytess.jpeg", im2_arr)

                elif (dim1[-1] != 11.0):
                    box_coord = [dim1[0], dim1[1], dim1[2], dim1[3]]
                    digit_lst.append([dim1[0], dim1[1], dim1[2], dim1[3], dim1[-1]])
                    box_lst.append(box_coord)
                    center_lst.append(get_center(box_coord))

        X = np.array(center_lst)
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X)
        y_kmeans = kmeans.predict(X)

        mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        if kmeans.cluster_centers_[1][0] > kmeans.cluster_centers_[0][0]:
            lst = mydict[1]
        else:
            lst = mydict[0]


        tmp_lst = []

        for ind in lst:
            tmp_lst.append(digit_lst[ind])

        tmp_lst = sorted(tmp_lst)
        price = [str(int(item[-1])) for item in tmp_lst]
        price = "".join(price[:-2]) + "." + price[-2] + price[-1]

        org = (int(dim[0]), int(dim[3]))
        org_bottom = (int(dim[0]), int(dim[3])+70)

        try:
            res = re.sub("[^0-9]", "", res)
            print(res)
            print(price)
            im_fin = cv2.putText(img = im_fin, text=res, org =org, fontFace =font, fontScale = fontScale, color =color, thickness =thickness)
            im_fin = cv2.putText(img = im_fin, text=price , org =org_bottom, fontFace =font, fontScale = fontScale, color =color, thickness =thickness)

        except:
            print("error")

    cv2.imwrite("./displ_res/im1_ocr_pytess{}.jpeg".format(str(time.time())), im_fin)
    plt.close('all')

lstf = glob.glob("./store/*")
for fl in lstf:
    try:
        analyze(fl)
    except Exception as e:
        print(e)
