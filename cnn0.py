
# libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import cv2
from sklearn.model_selection import train_test_split

# constants
img_dir = 'data/images'
data = pd.read_csv('legend.csv')
images = data["image"]
y = data['emotion']
y = [i.lower() for i in y]

# image resizer
def image_reader(location):
    img = cv2.imread(location, 0)
    img = cv2.resize(img, (50, 50))
    return img

emotion = set(y)
emotion = list(emotion)
print(emotion)
y = [emotion.index(i) for i in y]

# input data
x = []
for img in range(len(images)):
    location = os.path.join(img_dir, images[img])
    x.append(image_reader(location))
x = np.array(x)
x = x.reshape(13690, 50, 50, 1)
y = np.array(y)
print('Completed')

# model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                        input_shape=(50, 50, 1)))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x1, x2, y1, y2 = train_test_split(x, y, test_size = 0.3)

epochs = 100
batch_size = 128

history = model.fit(x1, y1, epochs=epochs,
          validation_data=(x2, y2), shuffle=True, verbose=1)

test_data = cv2.imread('test.jpg', 0)
test_data = cv2.resize(test_data, (60, 50))
test_data.resize(1, 50, 50, 1)

predictions = model.predict(test_data)

emotion[predictions.argmax()]

