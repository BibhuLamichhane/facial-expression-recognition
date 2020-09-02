import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
# DATA COLLECTION

train_dir = 'images/train'
input_shape = (48, 48, 1)

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

x1 = []
y1 = []
x2 = []
y2 = []
for emotion in range(len(emotions)):
    path = os.path.join(train_dir, emotions[emotion])
    data = os.listdir(path)
    print(f'Started the collection of the emotion "{emotions[emotion].upper()}"')
    for d in data:
        img = cv2.imread(os.path.join(path, d), 0)
        x1.append(img)
        y1.append(emotion)
    print(f'Emotion {emotions[emotion]} DONE')
print('Data collection complete')


x1 = np.array(x1)
y1 = np.array(y1)

x1 = x1.reshape(28821, 48, 48, 1)
y1 = tf.keras.utils.to_categorical(y1, 7)

model = models.Sequential()

model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(len(emotions), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.fit(x1, y1, batch_size=128, epochs=100)

model.save('model.h5')
