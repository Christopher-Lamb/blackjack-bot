import cv2
import first_frame
import numpy as np


class ShoeWatcher:
    def __init__(self):
        self.shoe_done = False
        self.checking = True
        self.count = 0
        self.shoe_px = 0

    def main(self, frame):
        """Take shoe frame return bool when masked for super red return shoe done"""
        self.shoe_done = False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper_red = np.array([200, 255, 255])
        lower_red = np.array([0, 200, 200])

        # print(frame[4:8, 10:14])
        # frame[4:8, 10:14] = [0, 0, 0]

        mask = cv2.inRange(hsv, lower_red, upper_red)
        number_of_white_pix = np.sum(mask == 255)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if number_of_white_pix >= 250 and self.checking:
            self.count = self.count + 1
        self.shoe_px = number_of_white_pix

        # cv2.imshow("shoe_watcher", frame)
        # cv2.imshow("mask", mask)

        # Needs 10 reds above 250 to determine new shoe
        if self.count >= 10:
            self.shoe_done = True
            self.checking = False
            self.count = 0
        # Once red dips below 100 px we Reset the Shoe Watcher to detect the next shoe
        if number_of_white_pix <= 100:
            self.checking = True

        y, x, d = frame.shape
        display = np.zeros((y, x * 3, 3), np.uint8)
        display[:y, :x] = frame
        display[:y, x : x * 2] = mask
        display[:y, x * 2 : x * 3] = result

        mask = cv2.resize(mask, (0, 0), fx=5, fy=5)
        # cv2.imshow("Shoe frame", display)

        # Return shoe done every frame
        return self.shoe_done


if __name__ == "__main__":
    red_upper = np.uint8([[[255, 255, 255]]])
    red_lower = np.uint8([[[21, 24, 192]]])
    # Convert red color to red HSV
    hsv_red_upper = cv2.cvtColor(red_upper, cv2.COLOR_BGR2HSV)
    hsv_red_lower = cv2.cvtColor(red_lower, cv2.COLOR_BGR2HSV)

    print(hsv_red_upper, hsv_red_lower)
