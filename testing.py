import cv2
import numpy as np
from tensorflow.keras.models import load_model
import screen_brightness_control as sbc
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time

background = None

# getting the audio device of the system
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
print(volume.GetMasterVolumeLevel())
print(volume.GetVolumeRange())


def running_avg(img, weight):
    """
    for running average calculation of background
    :param img: current background image
    :param weight: weight for the current image
    """
    global background
    if background is None:
        background = img.copy().astype(np.float)
        return

    cv2.accumulateWeighted(img, background, weight)


def segment_hand(img, threshold=25):
    """
    for hand segmentation from the image
    :param img: input image
    :param threshold: value of threshold for segmentation
    :return: thresholded image
    """
    global background
    diff = cv2.absdiff(background.astype(np.uint8), img)
    threshold_image = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    _, contours = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return None
    return threshold_image


Weight = 0.5  # weight for running average calculation of background

# read the video for which hand gestures will be controlling the commands
cap2 = cv2.VideoCapture('Harry.Potter.and.the.Half.Blood.Prince.2009.1080p.BrRip.x264.YIFY.mp4')
frames = []
curr_frame = 0
play = True

num_frames = 0
num_images = 0
prev_frame_time = 0
new_frame_time = 0
m = None
r = False
prediction = ''

# dictionary for mapping integer labels to their actual labels
d = {0: 'ClosingFist',
     1: 'LeftSwipe',
     2: 'OpeningFist',
     3: 'RightSwipe',
     4: 'SwipeDown',
     5: 'SwipeUp',
     6: 'ThumbsDown',
     7: 'ThumbsUp'}

# load the saved model
model = load_model('dynamic_hand_gesture_recognition.h5')

i = 0
print('Reading Video.....')
while cap2.isOpened():
    ret2, frame2 = cap2.read()
    if ret2:
        frame2 = cv2.resize(frame2, (480, 480))
        frames.append(frame2)
        i += 1
        if i == 5000:
            break
print('Done...')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    k = cv2.waitKey(1)
    if ret:
        frame = cv2.flip(frame, 1)
        roi = frame[0:320, 320:640]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.rectangle(frame, (320, 0), (640, 320), (0, 255, 0), 1)

        # check if num of frames are less than 200 or not
        # if yes then use running average function for background calculation
        if num_frames < 200:
            cv2.putText(frame, 'For first 200 frame do not show any hand gesture', (0, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            running_avg(gray, Weight)

        else:
            # get the segmented hand
            hand = segment_hand(gray)
            if hand is None:
                cv2.putText(frame, 'No hand gesture showed up on the screen', (0, 350), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0))
                if play:
                    curr_frame = min(curr_frame + 1, len(frames))
            else:
                cv2.imshow('hand', hand)

                # press q button to start saving images for the gesture in a matrix
                if k == ord('q'):
                    r = True
                    m = np.zeros(shape=(30, 128, 128, 1))

                if r:
                    if num_images % 5 == 0 and num_images <= 150:
                        hand = cv2.resize(hand, (128, 128))
                        hand = np.expand_dims(hand, axis=-1)
                        m[num_images // 5, :, :, :] = hand
                    num_images += 1

                    if num_images == 150:
                        r = False
                        num_images = 0

                        # get the predictions
                        m = np.expand_dims(m, axis=0)
                        prediction = d[np.argmax(model.predict(m), axis=1)[0]]
                        print(prediction)

                        # control various functions using predicted hand gesture
                        if prediction == 'ClosingFist':
                            # for pausing the video
                            curr_frame = curr_frame
                            play = False

                        elif prediction == 'OpeningFist':
                            # for playing the video
                            curr_frame = min(curr_frame + 1, len(frames))
                            play = True

                        elif prediction == 'LeftSwipe':
                            # for moving backwards in the video by 5 sec
                            curr_frame = max(0, curr_frame - 300)

                        elif prediction == 'RightSwipe':
                            # for moving forwards in the video by 5 sec
                            curr_frame = min(curr_frame + 300, len(frames))

                        elif prediction == 'SwipeUp':
                            # for increasing the brightness
                            current_brightness = sbc.get_brightness()
                            sbc.set_brightness(min(100, current_brightness + 10))
                            curr_frame = min(curr_frame + 1, len(frames))

                        elif prediction == 'SwipeDown':
                            # for decreasing the brightness
                            current_brightness = sbc.get_brightness()
                            sbc.set_brightness(max(0, current_brightness - 10))
                            curr_frame = min(curr_frame + 1, len(frames))

                        elif prediction == 'ThumbsUp':
                            # for increasing the volume
                            currentVolumeDb = volume.GetMasterVolumeLevel()
                            print(currentVolumeDb)
                            volume.SetMasterVolumeLevel(min(0, currentVolumeDb + 5.0), None)
                            print(volume.GetMasterVolumeLevel())
                            curr_frame = min(curr_frame + 1, len(frames))

                        elif prediction == 'ThumbsDown':
                            # for decreasing the volume
                            currentVolumeDb = volume.GetMasterVolumeLevel()
                            print(currentVolumeDb)
                            volume.SetMasterVolumeLevel(max(-63.5, currentVolumeDb - 5.0), None)
                            print(volume.GetMasterVolumeLevel())
                            curr_frame = min(curr_frame + 1, len(frames))
                    else:
                        if play:
                            curr_frame = min(curr_frame + 1, len(frames))
                else:
                    if play:
                        curr_frame = min(curr_frame + 1, len(frames))
                cv2.putText(frame, prediction, (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            cv2.imshow('video', frames[curr_frame])

        # frame rate calculation
        new_frame_time = time.time()
        if new_frame_time - prev_frame_time != 0:
            fps = 1/(new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))

        cv2.imshow('frame', frame)
        cv2.imshow('roi', roi)
        num_frames += 1

        # press Esc for closing the web cam and video
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
