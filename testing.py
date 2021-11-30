import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

background = None


def running_avg(img, weight):
    global background
    if background is None:
        background = img.copy().astype(np.float)
        return

    cv2.accumulateWeighted(img, background, weight)


def segment_hand(img, threshold=25):
    global background
    # print(background.shape)
    # print(img.shape)
    diff = cv2.absdiff(background.astype(np.uint8), img)
    threshold_image = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    _, contours = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(contours))
    if contours is None:
        return None
    return threshold_image


Weight = 0.5
cap = cv2.VideoCapture(0)
num_frames = 0
num_images = 0
prev_frame_time = 0
new_frame_time = 0
m = None
r = False
prediction = ''
d = {0: 'ClosingFist',
     1: 'LeftSwipe',
     2: 'OpeningFist',
     3: 'RightSwipe',
     4: 'SwipeDown',
     5: 'SwipeUp',
     6: 'ThumbsDown',
     7: 'ThumbsUp'}
model = load_model('hand_gesture_recognition_v3 (1).h5')

while cap.isOpened():
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if ret:
        frame = cv2.flip(frame, 1)
        roi = frame[0:320, 320:640]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.rectangle(frame, (320, 0), (640, 320), (0, 255, 0), 1)
        if num_frames < 200:
            cv2.putText(frame, 'For first 200 frame do not show any hand gesture', (0, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            running_avg(gray, Weight)

        else:
            hand = segment_hand(gray)
            if hand is None:
                cv2.putText(frame, 'No hand gesture showed up on the screen', (0, 350), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0))
            else:
                cv2.imshow('hand', hand)
                if k == ord('q'):
                    r = True
                    m = np.zeros(shape=(30, 128, 128, 1))

                if r:
                    if num_images % 5 == 0 and num_images <= 150:
                        hand = cv2.resize(hand, (128, 128))
                        # print(hand.shape)
                        hand = np.expand_dims(hand, axis=-1)
                        # hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
                        m[num_images // 5, :, :, :] = hand
                    num_images += 1

                    if num_images == 150:
                        r = False
                        num_images = 0
                        m = np.expand_dims(m, axis=0)
                        print(np.argmax(model.predict(m), axis=1))
                        prediction = d[np.argmax(model.predict(m), axis=1)[0]]
                        print(prediction)
                cv2.putText(frame, prediction, (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

        # new_frame_time = time.time()
        # fps = 1/(new_frame_time - prev_frame_time)
        # prev_frame_time = new_frame_time
        # fps = int(fps)
        # fps = str(fps)
        # cv2.putText(frame, fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

        cv2.imshow('frame', frame)
        cv2.imshow('roi', roi)
        num_frames += 1

        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
