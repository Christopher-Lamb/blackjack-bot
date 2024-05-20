import cv2
import torch
import first_frame
import numpy as np
from collections import Counter


# NOTE:
# We will take in dealer card frames and pixel cues
# We will detect a card change above 148 pixels for the first card
# This will set the presedant for the all cards after it
# When a cards pixel value reaches the presedant we will detect it.
# We will detect until we reach a confident card / 10/x times recorded

#  ==> this will make running the model more simple on my computer
#  ==> Also we might split dealer cards and player cards for different frames so that it stayes closer to real time


# Return Dealer Cards and is hand_done
# Return the first cards as well
class DealerCards:
    """Takes"""

    # default_precedant = 148

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.model = self.load_model()
        # self.started = False
        self.pixel_precedent = 150
        self.frame_list = []
        self.count = 0
        self.last_checked = 0
        # Add card once detected
        # Go off of the card_bool_list
        self.hand_done = False
        self.dealer_cards = []
        self.current_card = []
        self.last_index = 0
        # self.card_list = []
        self.total = 0
        self.card_bool_list = [False, False, False, False, False]
        self.px_list = []
        self.facedown_card = False

        # self.prev_pixels = [0, 0, 0, 0, 0]
        self.classes = self.model.names
        self.dealer_start = False
        self.device = "mps"

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """

        # path = "./yolov5"
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            "/Users/illmonk/Documents/python_scripts/blackjack-bot/yolov5/runs/train/exp56/weights/best.pt",
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
            # print("======found======\n", self.class_to_label(labels[i]))
            # print(f"{round(row[4].item() * 100,2)}%", "\n============")
            if row[4] >= 0.6:
                detected_list.append(self.class_to_label(labels[i]))
        return detected_list

    def check_card(self, index, px):
        """Check the vaule of the card with the model"""
        card_spot = self.frame_list[index]
        # cv2.imshow("CHECKCARD", card_spot)
        card_spot_gray = cv2.cvtColor(card_spot, cv2.COLOR_BGR2GRAY)
        h, w = card_spot_gray.shape
        white_color = card_spot_gray[5, h - 1]
        bg_color = int(self.pixel_precedent)
        if white_color > bg_color:
            bg_color = white_color

        background = np.zeros((128, 128, 3), np.uint8)
        background = cv2.rectangle(
            background,
            (0, 0),
            (background.shape[0], background.shape[1]),
            (int(bg_color), int(bg_color), int(bg_color)),
            -1,
        )
        w = (background.shape[1] - card_spot.shape[1]) // 2
        h = (background.shape[0] - card_spot.shape[0]) // 2

        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        background_gray[
            h : h + card_spot.shape[0], w : w + card_spot.shape[1]
        ] = card_spot_gray

        # thresh_frame = cv2.threshold(
        #     src=background_gray, thresh=127, maxval=bg_color, type=cv2.THRESH_BINARY
        # )[1]
        background_gray = cv2.resize(background_gray, (0, 0), fx=2, fy=2)

        # cv2.imshow(f"check_{index+1}", background_gray)

        # card_spot_resize = cv2.resize(background_gray, (0, 0), fx=2, fy=2)
        model_res = self.score_frame(background_gray)
        # print(model_res)
        results = self.format_results(model_res)
        # print(results)
        if len(results) != 0:
            print("checking", index)
            # print("PIXEL VALUE:", px)
            if px > self.pixel_precedent and index == 0:
                self.pixel_precedent = px
            # if self.can_check():
            self.current_card.append(results[0])
            occurance_dict = Counter(self.current_card)
            occurance_list = occurance_dict.most_common()
            name, num = occurance_list[0]

            if len(self.current_card) >= 15 or num >= 7:
                self.card_bool_list[index] = True
                self.dealer_cards.append(name)
                self.current_card = []
                # print(results[0])
                total, is_soft = self.calc_dealer_cards()
                self.handle_new_hand(total, is_soft)

    def handle_new_hand(self, total, is_soft):
        """Take total and is_soft and clear or coninue the and variables accordingly"""
        if total == 17 and is_soft:
            return
        elif total < 17:
            return
        if total >= 17:
            # Reset variables
            self.hand_done = True
            # print("Dealer TOTAL", total)

    def calc_dealer_cards(self):
        """Get the value of the dealer cards and check if 17 or soft 17"""
        total = 0
        # dealer_cards = ["A", "6", "K","6"]
        is_soft = False
        ace_count = 0
        for card in self.dealer_cards:
            if card in ["J", "Q", "K"]:
                card = "10"
            if card == "A":
                card = "11"
                ace_count = ace_count + 1

            total = total + int(card)

        for ace in range(ace_count):
            if total > 21:
                ace_count = ace_count - 1
                total = total - 10
        self.total = total
        if ace_count == 1:
            is_soft = True

        # print("dealer_amnt:", total, "\nis_soft:", is_soft)
        return (total, is_soft)

    # We want to get the 4 only to render if 3 2 & 1 are True
    # So we check the point that when off
    # Splice the array from that point and get all bools before it
    # If one False dont check card only check card if all before are true
    # NOTE I think I always intended this functionality just forgot

    def pixel_checker(self, pixel_list):
        """Take in pixel list compare with previous let know which change over 100 pos [1] the exception"""
        # print(pixel_list - self.prev_pixels)
        bools_true = self.card_bool_list.count(True)

        for idx, px in enumerate(pixel_list):
            if px >= (self.pixel_precedent - 25):
                # If max px value for cards we are checking changes to bright cards
                if not self.card_bool_list[idx]:
                    # RUN MODEL TO VERIFY CARD GOT FOUND
                    # If results conclusive then self.card_bool_list[] = True
                    all_before = self.card_bool_list[:idx]
                    if all(flag == True for flag in all_before):
                        # print("\nCheck Card", idx + 1)
                        self.check_card(idx, px)

            elif px <= 100:
                if idx == 0 and bools_true == 2:
                    # print(px)
                    self.card_bool_list = [False, False, False, False, False]
                    self.dealer_cards = []
                    self.pixel_precedent = 150

                elif idx == 2 and bools_true >= 3:
                    self.card_bool_list = [False, False, False, False, False]
                    self.dealer_cards = []
                    self.pixel_precedent = 150
            # Special Cases
            if (
                px <= (self.pixel_precedent - 20)
                and idx == 4
                and self.card_bool_list[4]
                and pixel_list[0] <= 100
            ):
                self.card_bool_list[4] = False
            # MAYBE DELETE TRACKS WHETHER DEALER HAS A FACEDOWN CARD
            if pixel_list[2] > 125 and pixel_list[1] < 125 and bools_true == 1:
                self.facedown_card = True
            else:
                self.facedown_card = False

            # if bools_true == 0 and len(self.current_card) >= 1:
            #     self.dealer_start = True
            if (
                bools_true == 0
                and pixel_list[1] >= (self.pixel_precedent - 25)
                and pixel_list[2] >= (self.pixel_precedent - 25)
            ):
                self.dealer_start = True

    def can_check(self):
        """Will run before card is added to see if card was checked with x number frames"""
        diff = self.count - self.last_checked
        if diff > 10:
            self.last_checked = self.count
            return True
        else:
            return False

    def use_model(self, frame_list, pixel_list):
        """Record all dealer cards in order passing information back to the GAME.py so we can play"""
        # FIRST: Know when to record a card.

        # Every pass through trigger hand done
        self.hand_done = False
        self.frame_list = frame_list
        c1, c2, c3, c4, c5 = pixel_list
        self.px_list = pixel_list
        self.dealer_start = False

        # total_w = 0
        # total_h = 0

        # for frame in frame_list:
        #     h, w, _ = frame.shape
        #     total_w = total_w + w
        #     if h > total_h:
        #         total_h = h
        # prev_w = 0
        # total_frame = np.zeros((total_h, total_w + 25, 3), np.uint8)

        # for frame in frame_list:
        #     h, w, _ = frame.shape
        #     total_frame[:total_h, prev_w + 5 : prev_w + w + 5] = frame
        #     prev_w = prev_w + w + 5

        # cv2.imshow("Dealer Card Watcher", total_frame)
        # print("Dealer Card Bools:",self.card_bool_list)
        # print("",self.dealer_cards)
        # If less than this value then no dealer card end of hand
        # if c1 >= 150:

        # print(pixel_list)
        self.pixel_checker(pixel_list)
        self.count = self.count + 1

        return (
            self.hand_done,
            self.dealer_cards,
            self.facedown_card,
            self.dealer_start,
        )


def get_frame(frame, str, dict):
    """Get frame from coords"""
    img = {}
    try:
        img = frame[
            dict[str][0][1] : dict[str][1][1],
            dict[str][0][0] : dict[str][1][0],
        ]
    except TypeError:
        pass

    return img


def coords_to_pixels(frame, coords, dict):
    """Take a list of coords turn into pixels"""
    pixels = []
    for coord in coords:
        color = frame[
            dict[coord][1] : dict[coord][1] + 1,
            dict[coord][0] : dict[coord][0] + 1,
        ]
        color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pixels.append(color[0][0])
    return pixels


def multiple_get_frames(frame, list_str, dict):
    """Process multiple get frames and return list of frames"""
    frame_list = []
    for string in list_str:
        new_frame = get_frame(frame, string, dict)
        frame_list.append(new_frame)

    return frame_list


if __name__ == "__main__":
    dealercards = DealerCards()
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(self.video_path)
    count = 0
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        # prev_frame = frame[:]

        if count == 0:
            main_dict = first_frame.get(frame)
        else:
            dealer_card_frames = multiple_get_frames(
                frame,
                ["one_card", "two_card", "three_card", "four_card", "five_card"],
                main_dict,
            )
            dealer_card_pixels = coords_to_pixels(
                frame,
                [
                    "one_card_active",
                    "two_card_active",
                    "three_card_active",
                    "four_card_active",
                    "five_card_active",
                ],
                main_dict,
            )
            (
                hand_done,
                dealer_cards,
                facedown_card,
                dealer_start,
            ) = dealercards.use_model(dealer_card_frames, dealer_card_pixels)
            # members = [
            #     attr
            #     for attr in dir(cardslot)
            #     if not callable(getattr(cardslot, attr)) and not attr.startswith("__")
            # ]
            print("\n\n\n\n\n")
            print("Dealer_card", dealercards.dealer_cards)
            for k, v in dealercards.__dict__.items():
                if k in ["model", "classes", "device", "frame_list"]:
                    continue
                print(k, v)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count = count + 1
