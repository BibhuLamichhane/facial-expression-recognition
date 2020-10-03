from tensorflow.keras.models import load_model
import cv2
import os


class EmotionPredictor:
    def predict(self, img):
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        if img in os.listdir():
            # load model
            mode = load_model('model.h5')

            # image processing
            img = cv2.resize(img, (48, 48))
            img = img.reshape(1, 48, 48, 1)
            img = img.astype('float64')

            # prediction
            predictions = mode.predict(img)
            output = []
            for prediction in range(len(predictions[0])):
                text = f'{emotions[prediction].upper()}: {(predictions[0][prediction] * 100).round()}'
                output.append(text)
            return output


if __name__ == '__main__':
    predict = EmotionPredictor()
    image = input("Enter file name with extension(jpg, png, etc)")
    predict.predict(image)
