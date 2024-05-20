import cv2
import torch
import numpy as np
from Timer import Timer
import math
import time
import first_frame
from collections import Counter

# Load the model
# Be able to pass the frame and get card value


class CardSlot:
    """Takes"""

    def __init__(self):
        """Initalizes the cards slot model"""
        self.model = self.load_model()
        self.classes = self.model.names
        # self.CardTimer = Timer(0, 25)
        self.current_card_list = []
        self.this_turn_card_list = []
        # self.card_record_switch = True
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
        self.card_list_len = 0
        self.motion_window = 0
        self.motion_limitter_switch = False
        self.motion_limitter = False
        self.post_motion_cnt = 0
        self.results = []
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
            if row[4] >= 0.4:
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
        # cv2.imshow("RED", mask)
        print(number_of_white_pix)
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
        if number_of_white_pix >= 3000:
            self.movement_detected = True
        else:
            self.movement_detected = False
            # thresh_frame = cv2.resize(thresh_frame,(0,0),fx=3,fy=3)
            # cv2.imshow(f"DETECT MOVEMENT{number_of_white_pix}", thresh_frame)

    def add_card(self, card_name):
        """Add card to this_turn_card_list"""
        self.this_turn_card_list.append(card_name)
        # If we reach our detecting limit add card to running list but for now just print
        self.shoe_ended = False
        self.current_card_list = []
        self.frame_cnt = 0
        self.is_detecting = False
        self.movement_detected = False
        self.motion_window = 0
        self.current_card_list = []

    def blur_frame(self, frame):
        """Prepare frame by blurring and"""
        prepared_frame = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=0)
        return prepared_frame

    def use_model(self, frame, dealer_stoppage, can_new_card, shoe_end):
        """Accepts a cardslot frame and stores relevant information in the class"""
        gray_frame = self.frame_adjuster(frame)
        prepared_frame = self.blur_frame(gray_frame)
        # print("\n\n\n\n\n\n")
        if self.prev_frame is None:
            # First frame; there is no previous one yet
            self.prev_frame = prepared_frame
            return []

        # WE are saying record one player card then stop
        # recording until dealer has one card or dealer_stoppage == True

        # Only start detecting if a card was found
        self.detect_movement(prepared_frame, self.prev_frame)
        if self.movement_detected and not self.is_detecting:
            model_res = self.score_frame(gray_frame)
            results = self.format_results(model_res)
            if len(results) > 0 and can_new_card and not dealer_stoppage:
                self.is_detecting = True

        # If can new card wait for movement to allow
        if shoe_end:
            self.detect_red(frame)
        results = []

        # We will now do it based on frames gathered
        # Then some how detect how many frames have passed till next card
        if self.motion_limitter:
            self.post_motion_cnt = self.post_motion_cnt + 1
            self.is_detecting = False
            if self.post_motion_cnt >= 40:
                self.motion_limitter = False
                self.post_motion_cnt = 0

        # When Motion detected allow for an amount of frames to be detected
        if self.is_detecting:
            self.motion_window = self.motion_window + 1
            model_res = self.score_frame(gray_frame)
            results = self.format_results(model_res)
            self.results = results

            # If not empty record
            if len(results) != 0:
                # If model returns with some card value record those
                self.record_in_timer(results)

            # THE PROBLEM
            # NOTE ONCE IT GETS GOING ITS RESETTING INSTEAD OF STOPPING AFTER 20 pluse
            # WE NNED A IS ACTIVE SWitcvh

            occurance_dict = Counter(self.current_card_list)
            occurance_list = occurance_dict.most_common()

            if len(occurance_list) > 0:
                name, num = occurance_list[0]
                if num >= 10 or self.motion_window >= 25:
                    self.add_card(name)
        # POST MOTION LIMITER
        if self.motion_limitter_switch:
            if len(self.this_turn_card_list) == self.card_list_len + 1:
                self.motion_limitter = True

        self.card_list_len = len(self.this_turn_card_list)
        self.prev_frame = prepared_frame

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
            card_list = cardslot.use_model(
                img, dealer_stoppage=False, can_new_card=True, shoe_end=False
            )
            # members = [
            #     attr
            #     for attr in dir(cardslot)
            #     if not callable(getattr(cardslot, attr)) and not attr.startswith("__")
            # ]
            # print("\n\n\n\n\n")
            for k, v in cardslot.__dict__.items():
                if k in [
                    "model",
                    "classes",
                    # "cut_card_px",
                    "device",
                    "shoe_ended",
                    "prev_frame",
                ]:
                    continue
                print(k, v)
            cv2.imshow("Card Watcher", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count = count + 1
