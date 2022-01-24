from email import header
from multiprocessing.spawn import import_main_path
from random import sample
from sklearn.model_selection import train_test_split
import numpy as np
import tensorboard
import tensorflow.keras as keras
import tensorflow as tf
import os
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import csv

np.random.seed(42)

'''Fetch datasets'''
TRAIN_SET_PATH = 'Digit_Recognizer/Datasets/train.csv'
TEST_SET_PATH = 'Digit_Recognizer/Datasets/test.csv'
MODEL_SAVE_PATH = 'Digit_Recognizer/digit_model.h5'

df_train = pd.read_csv(TRAIN_SET_PATH)
X_raw, y_raw = df_train.drop(columns='label'), df_train['label']
X_raw = np.array(X_raw, dtype=np.float32)
X_raw = X_raw / 255.
X_raw = X_raw.reshape(-1, 28, 28, 1)

X_full, X_test, y_full, y_test = train_test_split(X_raw, y_raw, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, random_state=42)

'''Creating rotate images(deg = 45, -45)'''
def create_two_rotated_images(images, deg=[45, -45], return_ratio=2, reshape=False):
    images_length = len(images)
    np.random.seed(43)
    idx_0 = np.random.choice(images_length, images_length // return_ratio)
    np.random.seed(42)
    idx_1 = np.random.choice(images_length, images_length // return_ratio)

    rotated_img_0 = []
    rotated_target_0 = []
    rotated_img_1 = []
    rotated_target_1 = []

    y_train_ = np.array(y_train)

    for id_0, id_1 in zip(idx_0, idx_1):
        rotated_img_0.append(rotate(images[id_0], deg[0], reshape=reshape))
        rotated_target_0.append(y_train_[id_0])
        rotated_img_1.append(rotate(images[id_1], deg[1], reshape=reshape))
        rotated_target_1.append(y_train_[id_1])
    
    return (np.array(rotated_img_0), rotated_target_0), (np.array(rotated_img_1), rotated_target_1)

(rotated_45_img, rotated_45_target), (rotated_315_img, rotated_315_target) = create_two_rotated_images(X_train, deg=[20, -20])

enlarged_X_train = np.r_[X_train, rotated_45_img, rotated_315_img]
enlarged_y_train = np.r_[y_train, rotated_45_target, rotated_315_target]

'''Define callbacks'''
checkpoint_callback = keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
early_stopping_callback = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

'''Build the model'''

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, 5, input_shape=[28,28,1], activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = keras.optimizers.Adam()

# model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# history = model.fit(enlarged_X_train, enlarged_y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[checkpoint_callback, early_stopping_callback])
# model.save(MODEL_SAVE_PATH)

model = keras.models.load_model(MODEL_SAVE_PATH)

'''I see that the model make a mistake images'''
# y_pred = model.predict(X_test)
# y_pred_col = np.argmax(y_pred, axis=1)
# wrong_img_mask = (y_test != y_pred_col)
# wrong_img = X_test[wrong_img_mask].reshape(-1, 28, 28)

# true_label = np.array(y_test[wrong_img_mask])
# pred_label = y_pred_col[wrong_img_mask]

# fig, axes = plt.subplots(3, 10, figsize=(10, 3))
# plt.subplots_adjust(wspace=0.2)

# for i in range(30):
#     ax = axes.ravel()
#     plt.sca(ax[i])
#     plt.imshow(wrong_img[i], cmap='binary')
#     plt.axis('off')
#     plt.title(f'Pred: {pred_label[i]}, True: {(true_label[i])}', fontdict={'fontsize': 6})
# plt.show()

'''Looking learning curve'''

def plot_learning_curve(loss, val_loss):
    plt.plot(np.arange(20), loss, 'b.-', label='Training')
    plt.plot(np.arange(20), val_loss, 'r.-', label='Validation')
    plt.axis([0, 20, 0, 0.1])
    plt.legend(loc='best')

# plot_learning_curve(history.history['loss'], history.history['val_loss'])
# plt.show()

X_true = pd.read_csv(TEST_SET_PATH) / 255.
X_true = np.array(X_true).reshape(-1, 28, 28, 1)
y_true = model.predict(X_true)
y_true_label = np.argmax(y_true, axis=1)
a = np.arange(1, 28001)

out = np.c_[a, y_true_label]

OUTPUT_CSV_PATH = 'Digit_Recognizer/output.csv'
col = ['ImageId', 'Label']

df = pd.DataFrame(out, columns=col)
df.to_csv(OUTPUT_CSV_PATH, index=False)


