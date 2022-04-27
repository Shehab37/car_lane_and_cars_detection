import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import glob
import os

# used for thresholds tuning
from ipywidgets import interact_manual
from IPython.display import display

from moviepy.editor import VideoFileClip
from IPython.display import HTML
# from tqdm import tqdm


h = 720
w = 1280

src = np.float32([
    [210, 700],
    [570, 460],
    [705, 460],
    [1075, 700]
])

dst = np.float32([
    [400, 720],
    [400, 0],
    [w-400, 0],
    [w-400, 720]
])


# meters per pixel in y dimension, 8 lines (5 spaces, 3 lines) at 10 ft each = 3m
ym_per_pix = 3*8/720
xm_per_pix = 3.7/550  # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters
