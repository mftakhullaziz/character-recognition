# Preprocessing_Citra.py

import cv2
import numpy as np
import math
# Gambar 1
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = 6

# beberapa bener
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = 4

# success in Sample_Plat/27.jpeg
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 23
# ADAPTIVE_THRESH_WEIGHT = 5

# success in Sample_Plat/16.jpeg
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 15
# ADAPTIVE_THRESH_WEIGHT = -17

# Berhasil di Citra 10.jpeg
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -11

# Work in Sample_Plat/11.jpeg
# GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -11

# Hasil

# berhasil pada plat 1.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -2

# berhasil pada plat 2.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -3

# Berhasil pada plat 3.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.2 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
# ADAPTIVE_THRESH_BLOCK_SIZE = 39
# ADAPTIVE_THRESH_WEIGHT = -16

# Berhasil pada plat 4.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = 0

# Berhasil pada plat 5.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.2 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 27
# ADAPTIVE_THRESH_WEIGHT = -6

# Berhasil pada plat 6.jpeg dengan cropping PLATE_WIDTH_PADDING_FACTOR = 1.1 dan PLATE_HEIGHT_PADDING_FACTOR = 1.5
# GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# ADAPTIVE_THRESH_BLOCK_SIZE = 17
# ADAPTIVE_THRESH_WEIGHT = -1

GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 17
ADAPTIVE_THRESH_WEIGHT = 2

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)

    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 5)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    # kernel = np.ones((5, 5), np.uint8)
    # imgThresh = cv2.dilate(imgThresh, kernel, iterations=0)
    # imgThresh = cv2.erode(imgThresh, kernel, iterations=0)

    return imgGrayscale, imgThresh
# end function

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 1), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function










