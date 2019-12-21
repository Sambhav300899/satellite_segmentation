from generator import *
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    gen = generator(bs = 1, input_shape = (512, 512, 3))

    while True:
        data = next(gen)
        cv2.imshow("img", data[0][0])
        cv2.imshow("mask", data[1][0] * 255.0)
        cv2.waitKey(0)
