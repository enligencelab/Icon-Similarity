import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import KFold
import pickle

# %%
ref = pd.read_excel("data/REFERENCE.xlsx")
img_root = "data/img_1/"
photo_size = 100

# %%
fp = ref['file1'][1]
img = cv2.imread(img_root + fp)
img = cv2.resize(src=img, dsize=(photo_size, photo_size))
img = np.array(img >= 200, dtype=np.uint8) * 255
img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(src=img, ksize=5)
img = cv2.Laplacian(src=img, ddepth=cv2.CV_8U, ksize=3)

# %%
img = cv2.imread("data/tests/case2.png", 0)

# %%
counters, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
counters = [np.array(x)[:, 0, ::-1] for x in counters if x.shape[0] > 12]

# %% test case
for counter in counters:
    tmp1 = np.zeros((100, 100))
    tmp1[counter[:, 0], counter[:, 1]] = 1
    plt.imshow(tmp1)
    plt.show()
plt.imshow(img);plt.show()

# %%
def distance_of_curves(c1: np.array, c2: np.array):
    c1_extended = c1[:, np.newaxis, :]
    c2_extended = c2[np.newaxis, :, :]
    distance = np.sqrt(np.sum(np.square(c1_extended - c2_extended), axis=2))
    distance_max = np.max(np.min(distance, axis=1))
    if distance_max < 4:
        return "coincide", distance_max

    c1_p5, c1_p95 = round(c1.shape[0] * .05), round(c1.shape[0] * .95)
    c2_p5, c2_p95 = round(c2.shape[0] * .05), round(c2.shape[0] * .95)
    connected = [

    ]
    return distance
