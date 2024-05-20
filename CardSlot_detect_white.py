import cv2
import torch
import numpy as np
from Timer import Timer
import math
import time
import first_frame

# Load the model
# Be able to pass the frame and get card value


class CardSlot:
    """Takes"""

    def __init__(self):
        """Initalizes the cards slot model"""
        self.model = self.load_model()
        self.classes = self.model.names
        self.CardTimer = Timer(0, 25)
        self.current_card_list = []
        self.this_turn_card_list = []
        self.card_record_switch = True
        # HOW MANY FRAMES TO DETECT ON
        self.frame_span = 4
        self.is_detecting = False
        self.frame_cnt = 0
        self.break_cnt = 60
        self.break_bool = False
        self.card_slot_px = 0
        self.cut_card_px = 0
        self.shoe_ended = False
        self.prev_frame = None
        self.movement_detected = False
        self.motion_px = 0
        self.motion_window = 0
        self.device = "mps"
        print("\n\nCard Slot Device:", self.device)

    def load_model(self):
        """ """
        # path = "./yolov5"
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            "/Users/illmonk/Documents/python_scripts/blackjack-bot/yolov5/runs/train/exp44/weights/best.pt",
        )
        return model

    def score_frame(self, frame):
        """ """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """ """
        return self.classes[int(x)]

    def format_results(self, results):
        """ """
        labels, cord = results
        detected_list = []
        n = len(labels)
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.7:
                detected_list.append(self.class_to_label(labels[i]))
        return detected_list

    def frame_adjuster(self, frame):
        """This will take the a frame and flip the right half so all numbers face up"""
        card_slot_frame = frame
        split = np.zeros((80, 80, 3), np.uint8)

        left_side = card_slot_frame[
            : card_slot_frame.shape[0], : card_slot_frame.shape[1] // 2
        ]
        right_side = card_slot_frame[
            : card_slot_frame.shape[0], card_slot_frame.shape[1] // 2 :
        ]
        h = round((80 - left_side.shape[0]) // 2)
        w = (80 - (left_side.shape[1] + right_side.shape[1])) // 2
        right_side = cv2.rotate(right_side, cv2.ROTATE_180)
        split[h : h + left_side.shape[0], w : w + left_side.shape[1]] = left_side
        split[
            h : h + left_side.shape[0],
            left_side.shape[1] + w : w + left_side.shape[1] + right_side.shape[1],
        ] = right_side
        # cv2.imshow("split", split)
        gray_split = cv2.cvtColor(split, cv2.COLOR_BGR2GRAY)

        return gray_split

    def detect_white(self, frame):
        """Accepts a frame and determines if images has a sufficent amount of white"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper_white = np.array([255, 55, 255])
        lower_white = np.array([0, 0, 200])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        number_of_white_pix = np.sum(mask == 255)
        self.card_slot_px = number_of_white_pix
        # cv2.imshow("SLOT FRAME", frame)
        # cv2.imshow("SLOT MASK", mask)
        # print(number_of_white_pix)
        # GOOD AT 600
        if number_of_white_pix >= 600:
            return (True, mask)
        else:
            return (False, mask)

    def card_slot_verifier(self, card_label_list):
        """Gets Current Card if 5 in a row make self.pass_card"""
        # print(self.CardTimer.timer_started)
        # card_label_list in format ["0"]
        # Start Timer if timer off

    def record_in_timer(self, card_label_list):
        """Pass array and record all the values into class"""
        for label in card_label_list:
            self.current_card_list.append(label)

    def detect_red(self, frame):
        """Detect when cut card is being sent through card slot"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper_red = np.array([200, 255, 255])
        lower_red = np.array([0, 200, 200])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        number_of_white_pix = np.sum(mask == 255)
        self.cut_card_px = number_of_white_pix
        if number_of_white_pix >= 200 and not self.shoe_ended:
            self.this_turn_card_list.append("CUTCARD")
            self.shoe_ended = True

    def detect_movement(self, prepared_frame, prev_frame):
        """Wait for movement on bottom of frame"""
        # If movement detected flip allow_check from movement
        diff_frame = cv2.absdiff(src1=prev_frame, src2=prepared_frame)
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)
        thresh_frame = cv2.threshold(
            src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY
        )[1]
        thresh_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)
        number_of_white_pix = np.sum(thresh_frame == 255)
        self.motion_px = number_of_white_pix
        if number_of_white_pix >= 6000:
            print(number_of_white_pix)
            self.movement_detected = True
            self.motion_window = 0
            # thresh_frame = cv2.resize(thresh_frame,(0,0),fx=3,fy=3)
            # cv2.imshow(f"DETECT MOVEMENT{number_of_white_pix}", thresh_frame)

    def blur_frame(self, frame):
        """Prepare frame by blurring and"""
        prepared_frame = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0)
        return prepared_frame

    def use_model(self, frame, break_override, can_new_card):
        """Accepts a cardslot frame and stores relevant information in the class"""
        gray_frame = self.frame_adjuster(frame)
        prepared_frame = self.blur_frame(gray_frame)
        # print("\n\n\n\n\n\n")
        if self.prev_frame is None:
            # First frame; there is no previous one yet
            self.prev_frame = prepared_frame
            return []

        can_detect, mask_frame = self.detect_white(frame)

        self.detect_red(frame)
        results = []
        if break_override:
            self.break_cnt = 1000
        # cv2.imshow("mask_frame", frame)
        

        # We will now do it based on frames gathered
        # Then some how detect how many frames have passed till next card
        if can_detect:
            model_res = self.score_frame(gray_frame)
            results = self.format_results(model_res)
            if not can_new_card:
                results = []
            # Establishes a break in record
            # print(len(results))
            # print(self.break_cnt)
            if len(results) == 0 and self.is_detecting:
                self.break_cnt = self.break_cnt + 1

            # checks for break
            if self.break_cnt >= 50:
                self.break_bool = True
            else:
                self.break_bool = False

            # Turns ON
            if len(results) != 0 and not self.is_detecting and self.break_bool:
                # print("START CNT")
                # We make sure there's a value and no active break point abd we start the breakpoint
                self.is_detecting = True
            # While ON
            if self.is_detecting:
                # This will count all frame after first frame with a card detected
                self.frame_cnt = self.frame_cnt + 1

            # NOTE if its not detected for frame span length then IF COMPLETE is never triggered
            # IF COMPLETE
            if self.frame_cnt >= self.frame_span:
                # print("COMPLETE")
                # If we reach our detecting limit add card to running list but for now just print
                self.shoe_ended = False
                val = max(set(self.current_card_list), key=self.current_card_list.count)
                self.this_turn_card_list.append(val)
                self.current_card_list = []
                self.is_detecting = False
                self.frame_cnt = 0
                self.break_cnt = 0
                # print(" ".join(self.this_turn_card_list))

            # If not empty record
            if len(results) != 0 and self.is_detecting:
                # If model returns with some card value record those
                self.record_in_timer(results)

        # If not enough white pixels allow for a card to be reorded next itteration of a card
        if not can_detect:
            # This will count when no cards detected
            # More specifically no white pixels
            self.break_cnt = self.break_cnt + 1
        # if hand_end:
        #     #Ignore the next cards that go through
        #     pass
        # Return Player Cards
        # player cards will always be the first two
        # Need a
        # print("This Turn:", self.this_turn_card_list)
        # print("CURRENT CARD:", self.current_card_list)
        # print("Break Cnt:", self.break_cnt)
        # print("Frame Cnt:", self.frame_cnt)
        # print("Break Bool:", self.break_bool)
        # Set Prev Frame
        self.prev_frame = prepared_frame
# 
        return self.this_turn_card_list

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(self.video_path)
    cardslot = CardSlot()
    count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        # prev_frame = frame[:]

        if count == 0:
            main_dict = first_frame.get(frame)
        else:
            img = frame[
                main_dict["card_slot"][0][1] : main_dict["card_slot"][1][1],
                main_dict["card_slot"][0][0] : main_dict["card_slot"][1][0],
            ]
            card_list = cardslot.use_model(img, break_override=False, can_new_card=True)
            # members = [
            #     attr
            #     for attr in dir(cardslot)
            #     if not callable(getattr(cardslot, attr)) and not attr.startswith("__")
            # ]
            print("\n\n\n\n\n")
            for k, v in cardslot.__dict__.items():
                if k in ["model","classes","cut_card_px","device","shoe_ended","prev_frame"]:
                    continue
                print(k,v)
            cv2.imshow("Card Watcher", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count = count + 1
