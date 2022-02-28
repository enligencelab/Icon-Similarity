import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import KFold
import pickle

# %%
ref = pd.read_excel("data/REFERENCE.xlsx")
img_root = "data/img_1/"
photo_size = 224

# %%
img1, img2, label = [], [], []


def cv_channel(fp):
    img = cv2.imread(img_root + fp)
    img = cv2.resize(src=img, dsize=(photo_size, photo_size))
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(src=img, ksize=9)
    img = cv2.Laplacian(src=img, ddepth=cv2.CV_8U, ksize=5)
    img = np.array(img >= 200, dtype=np.uint8) * 255
    return img


for f1, f2, sim_ in zip(ref['file1'], ref['file2'], ref['similar']):
    img1.append(cv_channel(f1))
    img2.append(cv_channel(f2))
    label.append(sim_)

# %%
img1 = np.array(img1, dtype=np.float32)[:, :, :, np.newaxis] / 255
img2 = np.array(img2, dtype=np.float32)[:, :, :, np.newaxis] / 255
label = np.array(label, dtype=np.float32)[:, np.newaxis]

# %%
kf = KFold(shuffle=True, random_state=60111)
dataset = []
rs = np.random.RandomState(seed=60111)
for train_idx, valid_idx in kf.split(label):
    rs.shuffle(train_idx), rs.shuffle(valid_idx)
    x1_train, x1_valid = img1[train_idx], img1[valid_idx]
    x2_train, x2_valid = img2[train_idx], img2[valid_idx]
    y_train, y_valid = label[train_idx], label[valid_idx]
    dataset.append([train_idx, valid_idx, x1_train, x2_train, y_train, x1_valid, x2_valid, y_valid])

# %%
with open(f"raw/dataset_filter_{photo_size}_{photo_size}", "wb") as f:
    pickle.dump(dataset, f)
