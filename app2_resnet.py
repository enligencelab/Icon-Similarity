from tensorflow.keras.applications import ResNet50V2
from tensorflow_addons.metrics import hamming_distance
import pickle

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np

# %% data
photo_size = 100
with open(f"raw/dataset_filter_{photo_size}_{photo_size}", "rb") as f:
    dataset = pickle.load(f)
validation = pd.DataFrame(columns=['Ground Truth', 'Estimated'])
diary = []

# %%
input_L = Input(shape=(photo_size, photo_size, 3))
feature_L = ResNet50V2(weights='imagenet', include_top=False, input_shape=(100, 100, 3), input_tensor=input_L)
for layer in feature_L.layers:
    layer._name += '_L'
input_R = Input(shape=(photo_size, photo_size, 3))
feature_R = ResNet50V2(weights='imagenet', include_top=False, input_shape=(100, 100, 3), input_tensor=input_R)
for layer in feature_R.layers:
    layer._name += '_R'

compare = tf.image.ssim(feature_L.output, feature_R.output, filter_size=1, max_val=1, k2=0.2)

stop_early = tf.keras.callbacks.EarlyStopping(patience=120, monitor='val_loss')
model = tf.keras.Model([input_L, input_R], compare)

def accuracy(y_true, y_predict):
    y_predict_float = tf.cast(y_predict > 0.5, tf.float32)
    equal_float = tf.cast(tf.equal(y_true, y_predict_float), tf.float32)
    return tf.reduce_mean(equal_float)

def contrastive_loss(y_true, y_predict):
    margin = 1
    square_predict = tf.square(y_predict)
    margin_square = tf.square(tf.maximum(margin - y_predict, 0))
    return (1 - y_true) * square_predict + y_true * margin_square

for k in range(5):
    save_best = tf.keras.callbacks.ModelCheckpoint(f"raw/app2_model_fold_{k}.h5", monitor="val_loss", save_best_only=True)
    model_0 = tf.keras.models.clone_model(model)
    model_0.compile(loss=contrastive_loss, metrics=[accuracy])
    _, _, x1_train, x2_train, y_train, x1_valid, x2_valid, y_valid = dataset[k]
    x1_train = np.repeat(x1_train, 3, -1)
    x2_train = np.repeat(x2_train, 3, -1)
    x1_valid = np.repeat(x1_valid, 3, -1)
    x2_valid = np.repeat(x2_valid, 3, -1)
    history_ = model_0.fit([x1_train, x2_train], y_train, validation_data=([x1_valid, x2_valid], y_valid),
                           callbacks=[stop_early, save_best], epochs=1000, batch_size=32)
    y_valid_hat = model_0.predict([x1_valid, x2_valid])[:, np.newaxis]

    validation_ = pd.DataFrame(data=np.hstack([y_valid, y_valid_hat]), columns=['Ground Truth', 'Estimated'])
    validation = pd.concat([validation, validation_])
    diary.append(history_.history)

validation.to_excel("raw/app2_validation.xlsx", index=False)
with open("raw/app2_training_history", "wb") as f:
    pickle.dump(diary, f)
