import cv2
import first_frame
import numpy as np


# NOTE We just need to be externally turned off and internally turned on
class NewCardWatcher:
    def __init__(self):
        self.new_card = False
        self.prev_frame = None
        self.new_card_watcher_px = 0

    def detect_white(self, img, prepared_frame, prev_frame):
        """Detect Signifigant Color Changes return bool"""
        diff_frame = cv2.absdiff(src1=prev_frame, src2=prepared_frame)
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)
        thresh_frame = cv2.threshold(
            src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY
        )[1]
        thresh_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

        number_of_white_pix = np.sum(thresh_frame == 255)
        self.new_card_watcher_px = number_of_white_pix
        if number_of_white_pix >= 600:
            # print(number_of_white_pix)
            # display = np.zeros((img.shape[0], img.shape[1] * 2, 3), np.uint8)
            # display[: img.shape[0], : img.shape[1]] = img
            # display[: img.shape[0], img.shape[1] : img.shape[1] * 2] = thresh_frame
            # cv2.imshow(f"Thresh{number_of_white_pix}", display)
            self.new_card = True

    def card_found(self):
        """If Game finds either a dealer or a player card we will filp new card to false"""
        self.new_card = False
    def seek_card(self):
        '''Be able to artificially trigger when we want to detect a new card'''
        self.new_card = False
        

    def main(self, frame):
        """Check if new card"""
        prepared_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)
        if self.prev_frame is None:
            # First frame; there is no previous one yet
            self.prev_frame = prepared_frame
            return

        self.detect_white(frame, prepared_frame, self.prev_frame)

        self.prev_frame = prepared_frame
        return self.new_card


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(self.video_path)
    watcher = NewCardWatcher()
    count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        # prev_frame = frame[:]

        if count == 0:
            main_dict = first_frame.get(frame)
        elif (count % 200) == 0:
            watcher.card_found()
        else:
            img = frame[
                main_dict["new_card_watcher"][0][1] : main_dict["new_card_watcher"][1][
                    1
                ],
                main_dict["new_card_watcher"][0][0] : main_dict["new_card_watcher"][1][
                    0
                ],
            ]
            new_card = watcher.main(img)
            print("\n\n\n\n\n", new_card)
            print(count, "\n\n\n")
            cv2.imshow("Card Watcher", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count = count + 1
