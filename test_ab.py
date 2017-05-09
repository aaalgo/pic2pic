#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('build/lib.linux-x86_64-2.7')
import numpy as np
import cv2
import _pic2pic


rgb = cv2.imread('lenna.png', cv2.IMREAD_COLOR)
rgb = rgb.astype(np.float32)
print("rgb:", rgb.shape)
rgb4 = np.expand_dims(rgb, 0)

L, ab, _ = _pic2pic.encode_lab(rgb4, 4)
L = L[0]
ab = ab[0]

print("L:", L.shape)
print("ab:", ab.shape)

ab_flat = np.reshape(ab, (-1, ab.shape[-1]))
#ab_dict = _pic2pic.ab_dict()

print(ab_dict)

#lab_big = np.zeros(L.shape[:2] + (,3), dtype=np.float32)
lab_small = np.zeros(ab.shape[:2] + (3,), dtype=np.float32)
print("ab_flat:", ab_flat.shape)
print("lab_small:", lab_small.shape)
lab_small[:, :, 1:] = np.reshape(np.dot(ab_flat, ab_dict), lab_small.shape[:2] + (2,))


H, W, _ = L.shape

lab = cv2.resize(lab_small, (W, H))
print("lab:", lab.shape)
lab[:, :, :1] = L

lab = lab.astype(np.float32)
rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

rgb *= 255

cv2.imwrite('output.png', rgb)

