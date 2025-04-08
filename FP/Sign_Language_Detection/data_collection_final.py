import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback

# Camera setup
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Directory setup
c_dir = 'A'
count = len(oss.listdir(f"C:\\FP\\Sign_Language_Detection\\AtoZ\\{c_dir}"))

offset = 15
step = 1
flag = False
suv = 0

# Create a blank white image
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("C:\\FP\\Sign_Language_Detection\\white.jpg", white)

skeleton1 = None  # Initialize to avoid undefined errors

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread("C:\\FP\\Sign_Language_Detection\\white.jpg")

        if hands:
            hand = hands[0]
            if isinstance(hand, dict) and 'bbox' in hand:
                x, y, w, h = hand['bbox']
                image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])

                handz, imz = hd2.findHands(image, draw=True, flipType=True)
                if handz:
                    hand = handz[0]
                    pts = hand['lmList']
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15

                    for t in range(0, 4):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                    for t in range(5, 8):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                    for t in range(9, 12):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                    for t in range(13, 16):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                    for t in range(17, 20):
                        cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)

                    cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

                    for i in range(21):
                        cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)

                    skeleton1 = np.array(white)
                    cv2.imshow("1", skeleton1)

        frame = cv2.putText(frame, f"dir={c_dir}  count={count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            break

        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):
                c_dir = 'A'
            flag = False
            count = len(oss.listdir(f"C:\\FP\\Sign_Language_Detection\\AtoZ\\{c_dir}\\"))

        if interrupt & 0xFF == ord('a'):
            flag = not flag
            if flag:
                suv = 0

        print("=====", flag)

        if flag and skeleton1 is not None:
            if suv == 180:
                flag = False
            if step % 3 == 0:
                print(f"Saving: {c_dir} - {count}")
                cv2.imwrite(f"C:\\FP\\Sign_Language_Detection\\AtoZ\\{c_dir}\\{count}.jpg", skeleton1)
                count += 1
                suv += 1
            step += 1

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()