#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:47:43 2018

@author: levi
"""

from img_drivers import mock_driver
import tracker
import utils
import numpy as np
import cv2

def ImageTranslation(img, ux, uy):
      M = np.identity(2)
      u = np.asarray([[ux, uy]])
      M = np.concatenate((M, u.T), axis=1)
      rows, cols = img.shape
      return cv2.warpAffine(img, M, (cols, rows))

def TestSimpleTranslation(ux, uy):
      driver = mock_driver.FromFiles('../images','bmp')
      img = driver.next()[0][0]
      img2 = ImageTranslation(img, ux, uy)
      my_tracker = tracker.TranslationTracker(img)
      idx = 10
      kp = my_tracker.keypoints[idx]
      shift = my_tracker.shifts[idx]
      computed_shift = my_tracker._find_shift(kp, shift, 
                                              my_tracker.kf_pyramid_xy_derivs[0], 
                                              my_tracker.kf_pyramid[0], img2)
      print('computed shift:', computed_shift)

def TestBigTranslation(ux, uy):
      driver = mock_driver.FromFiles('../images','bmp')
      img = driver.next()[0][0]
      my_tracker = tracker.TranslationTracker(img, 3)
      img2 = ImageTranslation(img, ux, uy)
      target_pyr = utils.build_pyramid(img2, my_tracker.pyramid_levels)
      idx = 100
      kp = my_tracker.keypoints[idx]
      shift = my_tracker.shifts[idx]
      computed_shift = my_tracker._find_shift_scale(kp, shift, target_pyr)
      print('computed shift:', computed_shift)
      print('initial kp:', my_tracker.keypoints[idx].pt)