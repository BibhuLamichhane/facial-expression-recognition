import cv2 as cv
import face_recognition
from predict import EmotionPredicter


def face_finder():
    predict = EmotionPredicter()
    cam = cv.VideoCapture(0)
    cv.namedWindow = 'Expression Recognizer'
    c = 0
    data = []
    while True:
        ret, f = cam.read()

        if not ret:
            break

        temp_frame = cv.resize(f, (0, 0), fx=0.25, fy=0.25)
        temp_frame = temp_frame[:, :, ::-1]

        face_loc = face_recognition.face_locations(temp_frame)
        for location in face_loc:
            top, right, bottom, left = location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            f = cv.rectangle(f, (left, top), (right, bottom), (150, 150, 150), 2)
            font = cv.FONT_HERSHEY_DUPLEX
            if len(data) > 0:
                cv.putText(f, data[0], (0, 20), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[1], (0, 40), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[2], (0, 60), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[3], (0, 80), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[4], (0, 100), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[5], (0, 120), font, 0.65, (0, 255, 0), 1)
                cv.putText(f, data[6], (0, 140), font, 0.65, (0, 255, 0), 1)
            if c == 10:
                face = f[top:bottom, left:right]
                face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                data = predict.predict(face)
                c = 0
            else:
                c += 1

        cv.imshow('', f)
        k = cv.waitKey(1)

        if k % 256 == ord('s'):
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    face_finder()
