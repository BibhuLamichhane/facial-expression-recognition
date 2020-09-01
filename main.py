from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# load model
mode = load_model('model.h5')

# load the image
img = cv2.imread('images/test/disgust/2275.jpg', 0)

# image processing
img = img.reshape(1, 48, 48, 1)
img = img.astype('float64')

# prediction
predictions = mode.predict(img)
for prediction in range(len(predictions[0])):
    print(f'{emotions[prediction].upper()}: {(predictions[0][prediction] * 100).round()}')
