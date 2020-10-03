import os
import cv2
import face_recognition
from tensorflow.keras.models import load_model


class EmotionPredictor:
    def __init__(self, image):
        self.image = cv2.imread(image)

    def find_faces(self):
        return face_recognition.face_locations(self.image)

    def emotion(self, img):
        emotions = ['angry', 'disgust', 'fear',
                    'happy', 'neutral', 'sad', 'surprise']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # load model
        mode = load_model('model.h5')

        # image processing
        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 1)
        img = img.astype('float64')

        # prediction
        predictions = mode.predict(img)
        output = ''
        x = float('-inf')
        for prediction in range(len(predictions[0])):
            if predictions[0][prediction] > x:
                x = predictions[0][prediction]
                output = emotions[prediction]
        return output

    def predict(self):
        locations = self.find_faces()
        img = self.image
        for location in locations:
            top, right, bottom, left = location

            face = img[top:bottom, left:right]
            emotion = self.emotion(face)
            print(emotion)

            font = cv2.FONT_HERSHEY_DUPLEX
            img = cv2.rectangle(
                img, (left, top), (right, bottom), (0, 0, 255), 2)
            img = cv2.rectangle(img, (left, bottom),
                                (right, bottom + 20), (0, 0, 255), -1)
            img = cv2.putText(img, emotion.upper(), (left + 3, bottom + 15),
                              font, 0.33, (255, 255, 255))

        cv2.imwrite("output.jpg", img)


if __name__ == '__main__':
    image = input("Enter the file path: ")
    predict = EmotionPredictor(image)
    predict.predict()
