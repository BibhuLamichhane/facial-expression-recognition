from tensorflow.keras.models import load_model
import cv2

# load the image
# img = cv2.imread('images/test/disgust/2275.jpg', 0)


class EmotionPredicter:
    def __int__(self):
        pass

    def predict(self, img):
        if img is not None:
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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
    predict = EmotionPredicter()
    predict.predict(None)

print('')