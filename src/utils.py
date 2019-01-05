import math
import cv2
import numpy as np

def build_pyramid(source, levels):
    pyramid = [source.copy()]
    for i in range(1,levels):
        img = cv2.pyrDown(pyramid[i-1])
        pyramid.append(img)
    return pyramid

def convert_to_kp(corners):
      kps = []
      for i in corners:
            kps.append(cv2.KeyPoint(i[0][0], i[0][1], 3))
      return kps

def find_features_to_track(source, max_corners=300, 
                           quality_level=0.01, min_distance=15, 
                           use_harris=True, block_size=5):
      corners = cv2.goodFeaturesToTrack(source, max_corners, 
                                        quality_level, min_distance,
                                        useHarrisDetector=use_harris)
      half_size = (block_size/2) + 1 if (block_size/2).is_integer() else math.ceil(block_size/2)
      criteria = (cv2.TermCriteria_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.03)
      corners = cv2.cornerSubPix(source, corners, (half_size, half_size), 
                                 (-1,-1), criteria)
      return convert_to_kp(corners)

def compute_xy_derivatives(source):
      deriv_x = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=5)
      deriv_y = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=5)
      return deriv_x, deriv_y

def compute_pyramid_xy_derivatives(pyramid):
      return [compute_xy_derivatives(x) for x in pyramid]

def compute_time_derivative(img_t0, img_t1):
      return cv2.subtract(img_t1, img_t0)

def stretch_matrixes(*V):
      vecs = [[]] * len(V)
      for i,v in enumerate(V):
            vecs[i] = v.reshape(-1,1)
      return np.concatenate(vecs, axis=1)

def compute_G_inv(x_deriv, y_deriv):
      if x_deriv.shape != y_deriv.shape:
            print('Error: shape mismatch for G_inv computation')
            return False
      A = stretch_matrixes(x_deriv, y_deriv)
      G = A.T.dot(A)
      return np.linalg.inv(G)

def compute_b(x_deriv, y_deriv, t_deriv):
      A = stretch_matrixes(x_deriv, y_deriv)
      t = stretch_matrixes(t_deriv)
      return A.T.dot(t)