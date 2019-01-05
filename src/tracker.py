import math
import cv2
import utils
import numpy as np

class TranslationTracker:
      def __init__(self, source = np.empty([2,2]), pyramid_levels=3):
            self.keyframes = []
            self.pyramid_levels = pyramid_levels
            self.kmargin = 0.02
            if source != np.empty([2,2]):
                  self.setKeyFrame(source)

      def setKeyFrame(self, keyframe):
            self.keyframes.append(keyframe)
            self.keypoints = utils.find_features_to_track(self.keyframes[-1])
            self.shifts = [[0,0]] * len(self.keypoints)
            self.is_trackable = [True] * len(self.keypoints)
            self.kf_pyramid = utils.build_pyramid(self.keyframes[-1], self.pyramid_levels)
            self.kf_pyramid_xy_derivs = utils.compute_pyramid_xy_derivatives(self.kf_pyramid)

      def track(self, target):
            self.shifts, _ = self._step(target, self.shifts)

      def _find_shift_scale(self, keypoint, shift,
                      target_pyramid,
                      window_sz=7):
            new_delta = shift
            lost = False
            for i in range(self.pyramid_levels-1, -1, -1):
                  scale = math.pow(2,i)
                  scaled_pos = cv2.KeyPoint(keypoint.pt[0] / scale, keypoint.pt[1] / scale, keypoint.size)
                  scaled_shift = np.asarray(new_delta) / scale
                  delta, lost = self._find_shift(scaled_pos, scaled_shift, self.kf_pyramid_xy_derivs[i], 
                                           self.kf_pyramid[i], target_pyramid[i])
                  #print('scale: %d, delta: %f, %f ---- [%f, %f]' % (i, delta[0], delta[1], delta[0] * scale, delta[1] * scale))
                  new_delta += delta * scale
                  if lost:
                        break
            return new_delta, lost

      def _find_shift(self, keypoint, 
                      shift, kf_xy_deriv, 
                      keyframe, target, min_delta_length=1e-4, max_rep=2000, window_sz=7):
            kf_x_deriv, kf_y_deriv = kf_xy_deriv
            #print(keypoint.pt)
            keyframe_kp_crop = cv2.getRectSubPix(keyframe, (window_sz, window_sz), keypoint.pt)
            kf_x_deriv_crop = cv2.getRectSubPix(np.float32(kf_x_deriv), (window_sz, window_sz), keypoint.pt)
            kf_y_deriv_crop = cv2.getRectSubPix(np.float32(kf_y_deriv), (window_sz, window_sz), keypoint.pt)
            g_inv = utils.compute_G_inv(kf_x_deriv_crop, kf_y_deriv_crop)
            total_delta = [0,0]
            count = 0
            delta_length = min_delta_length + 3
            lost = False
            while count < max_rep and delta_length >= min_delta_length:
                  #need inner loop for convergence
                  new_pos = np.asarray(keypoint.pt) + shift + total_delta
                  if self._is_outside_boundaries(new_pos, target.shape):
                        lost = True
                        break
                  target_kp_crop = cv2.getRectSubPix(target, (window_sz, window_sz), tuple(new_pos))
                  time_derivative = cv2.subtract(target_kp_crop, keyframe_kp_crop)
                  b = utils.compute_b(kf_x_deriv_crop, kf_y_deriv_crop, time_derivative)
                  delta = -g_inv.dot(b)
                  delta_length = np.linalg.norm(delta)
                  total_delta += delta.T[0]
                  #print(keypoint.pt, shift, total_delta)
                  count += 1
            #print(count, delta_length)
            return total_delta, lost

      def _step(self, target, shifts):
            target_pyramid = utils.build_pyramid(target, self.pyramid_levels)
            new_shifts = shifts.copy()
            lost_count = 0
            for i,feature in enumerate(self.keypoints):
                  lost = False
                  if (self.is_trackable[i]):
                        delta, lost = self._find_shift_scale(feature, shifts[i], target_pyramid)
                        new_shifts[i] += delta
                  if lost:
                        self.is_trackable[i] = False
                        lost_count += 1
            return new_shifts, lost_count

      def _is_outside_boundaries(self, pos, shape):
            margin = np.asarray(shape) * 0.0
            #print('position: (%f, %f), shape: [%d, %d], margin:[%d, %d]' % (pos[0], pos[1], shape[0], shape[1], margin[0], margin[1]))
            return ((pos[0] < 0 + margin[0]) 
                    or (pos[0] > shape[1] - margin[0]) 
                    or (pos[1] < 0 + margin[1]) 
                    or (pos[1] > shape[0] - margin[1]))