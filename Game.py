import cv2
import time
import datetime
import math
from Timer import Timer
from itertools import groupby
import numpy as np
import time
from collections import Counter
from datetime import datetime

# from DealerCards import DealerCards
import first_frame
from CardSlot import CardSlot
from DealerCards import DealerCards
from ShoeWatcher import ShoeWatcher
from BasicStrategy import BasicStrategy
from BlackjackGUI import BlackjackGUI
from NewCardWatcher import NewCardWatcher

# Note
"""Each game will begin with a play hand this will ensure that the timer is correct and we can accuratly pass through each boot cycle"""
"""Each Class for watching will be hooked up to switches for example
dealer_arr = DealerCardsClass() 
"""


class Game(Timer):
    """We will create a class that allows us to recieve important blackjack information and pass it on to actions we want to carry out
    It will also house the main frame while loop and split each frame and send it to it's corresponding class
    """

    betting_unit = 1

    def __init__(self, bankroll):
        # Card Slot
        self.CardSlot = CardSlot()
        self.length_at_done = 100
        self.player_card_list = []
        self.player_card_len = 100
        # Dealer Cards
        self.DealerCards = DealerCards()
        self.dealer_cards_len = 100
        self.hand_done = False
        self.dealer_cards = []
        # Shoe Watcher
        self.ShoeWatcher = ShoeWatcher()
        # New Card Watcher
        self.NewCardWatcher = NewCardWatcher()
        self.can_new_card = False
        # BasicSrategy
        self.BasicStrategy = BasicStrategy()
        self.BlackjackGUI = BlackjackGUI()
        # === === === === ===
        # WONG HALVES
        self.wong_running_cnt = 0
        self.wong_true_cnt = 0
        self.wong_bet_sizing = 0
        # THORP ULTIMATE
        self.thorp_running_cnt = 0
        self.thorp_true_cnt = 0
        self.thorp_bet_sizing = 0
        # === === === === ===
        self.CardTimer = Timer(0, 10)
        self.last_card_detected = ""
        # self.video_path = video_path
        self.frame_count = 0
        # self.player_cards = []
        self.position_dict = {}
        # GAME Exclusive
        self.deck = []
        self.shot_clock_px = 0
        self.game_gap = False
        self.clock_on = False
        self.can_place_bet = False
        self.can_do_action = False
        self.play_next_hand = False
        self.session_started = False
        # === HANDLE PLAY ===
        self.is_playing = True
        self.play_done = False
        self.splitting = False
        self.player_hand1 = []
        self.player_hand2 = []
        self.player_hand1_done = False
        self.player_hand2_done = False
        self.prev_player_card_len = 100
        self.player_action_list = []
        self.bet_placed = False
        self.betting_amount = 0
        self.last_played_frame = -2
        self.shoe_done = False
        self.insurance = False
        self.insurance_action = False
        self.player_hand1_total = 0
        self.player_hand2_total = 0
        self.dealer_stoppage = False
        self.last_removed = ""
        self.hand_proccessed = False
        self.preprocessed_hand = []
        self.outcome = ""
        self.outcome2 = ""
        self.current_bet_1 = 0
        self.current_bet_2 = 0
        self.bankroll = bankroll
        self.this_error = False
        self.hand_cnt = 0
        # self.facedown_card = False
        # self.facedown_switch = False
        Timer.__init__(self, 5, 00)

    def get_frame(self, frame, str):
        """Get frame from coords"""
        img = {}
        try:
            img = frame[
                self.position_dict[str][0][1] : self.position_dict[str][1][1],
                self.position_dict[str][0][0] : self.position_dict[str][1][0],
            ]
        except TypeError:
            pass

        return img

    def build_deck(self):
        """Lol this will build the deck"""
        deck_num = 8
        card_num = deck_num * 4
        deck = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "10", "10", "10", "11"]
        self.deck = []
        for num in range(0, card_num):
            for item in deck:
                self.deck.append(item)

    def multiple_get_frames(self, frame, list_str):
        """Process multiple get frames and return list of frames"""
        frame_list = []
        for string in list_str:
            new_frame = self.get_frame(frame, string)
            frame_list.append(new_frame)

        return frame_list

    def coords_to_pixels(self, frame, coords):
        """Take a list of coords turn into pixels"""
        pixels = []
        for coord in coords:
            color = frame[
                self.position_dict[coord][1] : self.position_dict[coord][1] + 1,
                self.position_dict[coord][0] : self.position_dict[coord][0] + 1,
            ]
            color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            pixels.append(color[0][0])
        return pixels

    def wong_halves_counting(self, card_val):
        """Implement the wong halves card counting system to keep true count"""
        if card_val in ["11", "10"]:
            self.wong_running_cnt = self.wong_running_cnt - 1
        elif card_val == "9":
            self.wong_running_cnt = self.wong_running_cnt - 0.5
        elif card_val in ["2", "7"]:
            self.wong_running_cnt = self.wong_running_cnt + 0.5
        elif card_val in ["3", "4", "6"]:
            self.wong_running_cnt = self.wong_running_cnt + 1
        elif card_val == "5":
            self.wong_running_cnt = self.wong_running_cnt + 1.5

        divisor = len(self.deck) / 52
        self.wong_true_cnt = round(self.wong_running_cnt / divisor, 2)
        self.BasicStrategy.set_true_count(self.wong_true_cnt)

    def thorp_ultimate_counting(self, card_val):
        """IMplement Thorp Ultimate Card Counting system"""
        if card_val == "2":
            self.thorp_running_cnt = self.thorp_running_cnt + 0.5
        elif card_val == ["3", "6"]:
            self.thorp_running_cnt = self.thorp_running_cnt + 0.6
        elif card_val == "4":
            self.thorp_running_cnt = self.thorp_running_cnt + 0.8
        elif card_val == "5":
            self.thorp_running_cnt = self.thorp_running_cnt + 1.1
        elif card_val == "7":
            self.thorp_running_cnt = self.thorp_running_cnt + 0.4
        elif card_val == "9":
            self.thorp_running_cnt = self.thorp_running_cnt - 0.3
        elif card_val == "10":
            self.thorp_running_cnt = self.thorp_running_cnt - 0.7
        elif card_val == "11":
            self.thorp_running_cnt = self.thorp_running_cnt - 0.9

        divisor = len(self.deck) / 52
        self.thorp_true_cnt = round((self.thorp_running_cnt / divisor), 2)

    def shot_clock_check(self, frame):
        """identify when a shot clock is in places"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper_green = np.array([130, 255, 255])
        lower_green = np.array([47, 153, 184])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        number_of_white_pix = np.sum(mask == 255)
        self.shot_clock_px = number_of_white_pix
        # cv2.imshow("MASK", mask)
        # cv2.imshow("FRAME", frame)

        if number_of_white_pix >= 40:
            self.clock_on = True
        else:
            self.clock_on = False

    # If just played restart timer
    def wanna_play(self):
        """Figure out if we want to play or not"""
        timer_fin, message = self.compare_time()
        if timer_fin:
            self.play_next_hand = True
        else:
            self.play_next_hand = False

        if self.wong_true_cnt >= 1.5 and len(self.deck) <= 290:
            self.play_next_hand = True

    def determine_bet_size(self):
        """Determine the bet size"""
        if self.wong_true_cnt < 1.5:
            self.betting_amount = 1
        elif self.wong_true_cnt <= 2:
            self.betting_amount = 2
        elif self.wong_true_cnt <= 2.5:
            self.betting_amount = 2
        elif self.wong_true_cnt <= 3:
            self.betting_amount = 3
        elif self.wong_true_cnt > 3:
            self.betting_amount = 3

    def game_eval(self, dealer_total, player_total, player_cards_len):
        """Take player total and Dealer total and determine a winner"""
        outcome = ""
        if player_cards_len == 2 and player_total == 21 and dealer_total != 21:
            outcome = "BLACKJACK"
        elif player_total > 21:
            outcome = "LOSE"
        elif player_total == dealer_total:
            outcome = "PUSH"
        elif dealer_total > 21:
            outcome = "WIN"
        elif player_total > dealer_total:
            outcome = "WIN"
        elif dealer_total > player_total:
            outcome = "LOSE"

        return outcome

    def main(self):
        """Handle main game loop"""
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture(self.video_path)
        count = 0
        while cap.isOpened():
            """Main Game Video Loop"""
            ret, frame = cap.read()
            if count == 0:
                # print("first_frame")
                self.position_dict = first_frame.get(frame)
                self.build_deck()
                self.BlackjackGUI.build_coords(
                    self.position_dict,
                    "hit_btn",
                    "stand_btn",
                    "double_btn",
                    "split_btn",
                    "dollar_chip",
                    "make_bet_btn",
                )
            elif (count % 1) == 0:
                # print("\n\n\n\n\n\n")
                # Timer Logic
                try:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
                    cv2.imshow("frame", small_frame)
                except:
                    print("WHATEVER THIS ERROR IS ")
                    if not self.this_error:
                        self.this_error = True
                        string = f"THIS ERROR\n"
                        file1 = open("game_data.txt", "a")
                        file1.write(string)
                        file1.close()
                    continue

                if not self.session_started:
                    self.play_next_hand = False
                    self.session_started = True
                    file1 = open("game_data.txt", "a")
                    file1.write("SESSION START\n")
                    file1.close()
                # print(self.upcard, self.hand_done, self.dealer_cards)

                # NOTE NEW CARD WATCHER
                new_card_frame = self.get_frame(frame, "new_card_watcher")
                new_card = self.NewCardWatcher.main(new_card_frame)
                self.can_new_card = new_card
                # NOTE NEW CARD WATCHER END

                # NOTE HANDLE SHOE
                shoe_frame = self.get_frame(frame, "shoe_deck")
                shoe_done = self.ShoeWatcher.main(shoe_frame)
                if shoe_done:
                    self.shoe_done = True
                # NOTE HANDLE SHOE END

                # NOTE SHOT CLOCK
                shot_clock_frame = self.get_frame(frame, "shot_clock")
                self.shot_clock_check(shot_clock_frame)
                # NOTE SHOT CLOCK END

                # Get All frame
                # print(self.position_dict)

                # Record the length at time of hand end then wait until the length increase by one
                # From here we will pop and record into the deck

                # NOTE HANDLE CARD_SLOT NOTE
                card_slot_frame = self.get_frame(frame, "card_slot")

                card_list = self.CardSlot.use_model(
                    card_slot_frame, self.dealer_stoppage, self.can_new_card, shoe_done
                )
                # HANDLES CARDSLOT READING DEALER LAST CARD
                if self.hand_done:
                    self.length_at_done = len(card_list)

                if len(card_list) == (self.length_at_done + 1):
                    card_list.pop()
                    # print(card_list)
                    self.NewCardWatcher.card_found()
                    self.CardSlot.this_turn_card_list = []
                    self.length_at_done = 100
                    self.player_card_len = 100
                    # time.sleep(2)
                if self.game_gap and self.clock_on:
                    # MAYBE REMOVE
                    self.NewCardWatcher.card_found()
                    self.CardSlot.this_turn_card_list = []
                    self.length_at_done = 100
                    self.player_card_len = 100

                self.player_card_list = card_list

                # Since cards are added one at a time when new card is added remove from deck
                # also convert cards to their respective values
                # REMOVE CARD_SLOT CARD FROM DECK ONE BY ONE
                # CARD SLOT CARD COUNT AREA
                if len(card_list) > self.player_card_len and self.hand_proccessed:
                    # run delete last card from deck
                    self.NewCardWatcher.card_found()
                    val = card_list[-1]
                    self.last_card_detected = val
                    if val in ["J", "Q", "K"]:
                        val = "10"
                    if val == "A":
                        val = "11"
                    self.deck.remove(val)
                    self.last_removed = val
                    self.wong_halves_counting(val)
                    self.thorp_ultimate_counting(val)

                self.player_card_len = len(card_list)

                # print(
                #     "\n\n\n\n\n\nCard_slot:",
                #     len(card_list),
                #     "\nCard_slot_Prev",
                #     self.player_card_len,
                # )

                # NOTE HANDLE CARD_SLOT END NOTE

                # NOTE DEALER CARDS NOTE
                dealer_card_frames = self.multiple_get_frames(
                    frame,
                    ["one_card", "two_card", "three_card", "four_card", "five_card"],
                )
                dealer_card_pixels = self.coords_to_pixels(
                    frame,
                    [
                        "one_card_active",
                        "two_card_active",
                        "three_card_active",
                        "four_card_active",
                        "five_card_active",
                    ],
                )
                (
                    hand_done,
                    dealer_cards,
                    facedown_card,
                    dealer_start,
                ) = self.DealerCards.use_model(dealer_card_frames, dealer_card_pixels)

                self.facedown_card = facedown_card
                if facedown_card:
                    pass
                if (facedown_card and not self.hand_proccessed) or (
                    self.can_do_action and not self.hand_proccessed
                ):
                    # Evaluate hand and deal with duplicates
                    time.sleep(2)
                    self.CardSlot.motion_limitter_switch = True
                    self.NewCardWatcher.card_found()
                    self.hand_proccessed = True
                    card_adding = ""
                    if len(self.CardSlot.current_card_list) > 0:
                        occurance_dict = Counter(self.CardSlot.current_card_list)
                        occurance_list = occurance_dict.most_common()
                        name, num = occurance_list[0]
                        card_adding = name
                    else:
                        card_adding = card_list[-1]

                    self.CardSlot.add_card(card_adding)
                    card_list = self.CardSlot.this_turn_card_list
                    occurance_dict = Counter(card_list)
                    occurance_list = occurance_dict.most_common()
                    formatted_cards = []
                    if len(occurance_list) == 1:
                        formatted_cards = [occurance_list[0][0], occurance_list[0][0]]
                    elif len(occurance_list) >= 2:
                        formatted_cards = [occurance_list[0][0], occurance_list[1][0]]

                    for card in formatted_cards:
                        self.last_card_detected = card
                        if card in ["J", "Q", "K"]:
                            card = "10"
                        if card == "A":
                            card = "11"
                        self.deck.remove(card)
                        self.last_removed = card
                        self.wong_halves_counting(card)
                        self.thorp_ultimate_counting(card)

                    self.CardSlot.this_turn_card_list = formatted_cards

                    self.player_action_list = [False, False]

                    self.player_card_list = formatted_cards
                    self.player_card_len = len(formatted_cards)

                # Add this back after done dealer cards
                if len(dealer_cards) > self.dealer_cards_len:
                    # run delete last card from deck
                    # if len(dealer_cards) == 1:
                    #     # If dealer stoppage False than we can look for our second card
                    #     self.dealer_stoppage = False

                    val = dealer_cards[-1]
                    self.last_card_detected = val
                    if val in ["J", "Q", "K"]:
                        val = "10"
                    if val == "A":
                        val = "11"
                    self.deck.remove(val)
                    self.last_removed = val
                    # RIGHT HERE send through card counter
                    self.wong_halves_counting(val)
                    self.thorp_ultimate_counting(val)

                    # self.NewCardWatcher.card_found()

                self.hand_done = hand_done
                if self.player_card_len == 2 and self.dealer_cards_len == 2:
                    card_list = []
                    for card in self.player_card_list:
                        if card in ["J", "Q", "K"]:
                            card = "10"
                        if card == "A":
                            card = "11"
                        card_list.append(card)
                    total, _ = self.BasicStrategy.get_total(card_list)
                    if total == 21:
                        self.hand_done = True

                self.dealer_cards = dealer_cards
                self.dealer_cards_len = len(dealer_cards)

                # print(
                #     "Dealer:",
                #     len(dealer_cards),
                #     "\nDealer Prev:",
                #     self.dealer_cards_len,
                #     "\n",
                #     len(self.deck),
                # )
                # print("Last Card:", self.last_card_detected)
                # print("Wong run true", self.wong_running_cnt, self.wong_true_cnt)
                # print("Thorp run true", self.thorp_running_cnt, self.thorp_true_cnt)

                # NOTE FINAL DEALER CARDS NOTE

                # So Dealer cards handles when hand is done
                # Card slot handles when cards start
                # If hand done we have to ignore one card slot card

                # If amount of dealer cards changes remove last card add to dekc
                if len(dealer_cards) > len(self.dealer_cards):
                    self.deck.remove(dealer_cards[-1])
                # NOTE
                # # DISPLAY

                # NOTE HANDLE HAND ACTIONS
                if self.hand_done:
                    self.game_gap = True
                    self.is_playing = False
                    self.bet_placed = False
                    self.hand_proccessed = False
                    self.CardSlot.motion_limitter_switch = False

                    # SHOE SWITCH GOES IN HERE RESTART SHOE IF FILPPED AFTER HAND_DONE
                    if self.shoe_done:
                        file1 = open("game_data.txt", "a")
                        file1.write("SHOE BREAK\n")
                        file1.close()
                        self.hand_cnt = 0
                        self.shoe_done = False
                        self.build_deck()
                        self.thorp_running_cnt = 0
                        self.thorp_true_cnt = 0
                        self.wong_running_cnt = 0
                        self.wong_true_cnt = 0

                self.determine_bet_size()
                if self.dealer_cards_len > 0:
                    self.game_gap = False
                else:
                    self.game_gap = True

                if self.game_gap and self.clock_on:
                    # Place bet
                    self.can_place_bet = True
                else:
                    self.can_place_bet = False

                if not self.game_gap and self.clock_on:
                    # Do Action
                    self.can_do_action = True
                else:
                    self.can_do_action = False

                # We have to determine if we want to play a game
                # How do we determine that
                # We want to play if the timer is up or if true count is up

                # if len(self.player_card_list) >= 2 and len(self.dealer_cards) > 0:

                #     )

                # NOTE HANDLE Place Bet

                # NOTE HANDLE PLAY FUNCTION
                if self.player_card_len == self.prev_player_card_len + 1:
                    # So everytime this triggers we want to add a false action to the player_action_list
                    self.player_action_list.append(False)

                if self.is_playing:
                    # Move our player accordingly
                    self.last_played_frame = self.frame_count
                    # Handle Movement
                    if (
                        self.can_place_bet
                        and not self.bet_placed
                        and len(self.player_card_list) == 0
                    ):
                        # print("PLACE BET")
                        self.BlackjackGUI.place_bet(self.betting_amount)
                        self.current_bet_1 = self.betting_amount
                        self.bet_placed = True
                        # self.bet_wait_frame = 0

                    if (
                        self.can_do_action
                        and not self.play_done
                        and self.hand_proccessed
                    ):
                        # HANDLE POSSIBLE INSURANCE AGAINST ACE BY DENYING ALL
                        if dealer_cards[0] == "A" and not self.insurance_action:
                            self.insurance = True
                        if self.insurance and not self.insurance_action:
                            self.BlackjackGUI.player_action("STAND")
                            self.insurance_action = True

                        # We want this to run once everytime
                        # If cards len increases by one run
                        # print("Do action")
                        if not self.insurance:
                            action = self.BasicStrategy.get_action(
                                self.dealer_cards[0], self.player_card_list
                            )
                            if action == "SPLIT":
                                self.splitting = True
                            if self.splitting:
                                if (
                                    not self.player_action_list[0]
                                    and not self.player_action_list[0]
                                ):
                                    # DO SPLIT ACTION HERE
                                    self.BlackjackGUI.player_action("SPLIT")
                                    self.player_hand1.append(self.player_card_list[0])
                                    self.player_action_list[0] = True
                                    self.player_hand2.append(self.player_card_list[1])
                                    self.current_bet_2 = self.current_bet_1
                                    self.player_action_list[1] = True
                                if not all(self.player_action_list):
                                    if not self.player_hand1_done:
                                        # Get false index
                                        index = self.player_action_list.index(False)
                                        self.player_hand1.append(
                                            self.player_card_list[index]
                                        )
                                        h1_action = self.BasicStrategy.get_action(
                                            self.dealer_cards[0],
                                            self.player_hand1,
                                            True,
                                        )

                                        if h1_action == "STAND":
                                            self.player_hand1_done = True
                                            self.player_action_list[index] = True
                                            self.player_hand1_total = (
                                                self.BasicStrategy.total
                                            )
                                            # No need for stand action let time run out
                                        else:
                                            self.player_action_list[index] = True
                                            self.BlackjackGUI.player_action(h1_action)
                                            if h1_action == "DOUBLE":
                                                self.player_hand1_done = True
                                                self.current_bet_1 = (
                                                    self.current_bet_1 * 2
                                                )

                                        # DO ACTION HERE
                                        # self.HandleGUI.handle_action(action)
                                    elif not self.player_hand2_done:
                                        index = self.player_action_list.index(False)

                                        self.player_hand2.append(
                                            self.player_card_list[index]
                                        )
                                        h2_action = self.BasicStrategy.get_action(
                                            self.dealer_cards[0],
                                            self.player_hand2,
                                            True,
                                        )
                                        self.player_action_list[index] = True
                                        # DO ACTION HERE
                                        self.BlackjackGUI.player_action(h2_action)
                                        # self.HandleGUI.handle_action(action)

                                        if h2_action == "STAND":
                                            self.player_hand2_done = True
                                            self.player_hand2_total = (
                                                self.BasicStrategy.total
                                            )
                                        if h2_action == "DOUBLE":
                                            self.current_bet_2 = self.current_bet_2 * 2

                            else:
                                # You initalize the first value and append so we work with two cards
                                # Because you only want one action out of it
                                if not self.player_action_list[0]:
                                    self.player_action_list[0] = True
                                    self.player_hand1.append(self.player_card_list[0])
                                if (
                                    not all(self.player_action_list)
                                    and not self.player_hand1_done
                                ):
                                    # self.player_action_list[1] = True
                                    index = self.player_action_list.index(False)
                                    self.player_hand1.append(
                                        self.player_card_list[index]
                                    )
                                    h1_action = self.BasicStrategy.get_action(
                                        self.dealer_cards[0], self.player_card_list
                                    )
                                    self.player_action_list[index] = True
                                    self.BlackjackGUI.player_action(action)
                                    if h1_action == "STAND":
                                        self.player_hand1_done = True
                                        self.player_hand1_total = (
                                            self.BasicStrategy.total
                                        )
                                    if h1_action == "DOUBLE":
                                        self.current_bet_1 = self.current_bet_1 * 2

                    self.insurance = False
                    # Handle split
                    # After each hand we want to determine if we want to play

                # The next frame that we are done playing restart the timer
                if (
                    not self.is_playing
                    and self.frame_count == self.last_played_frame + 1
                ):
                    self.init_timer()
                    # SAVE HAND PLAYED INFO
                    # total , _ = self.BasicStrategy.get_total()
                    # If cards busted or got skipped bc no player action was require
                    # we will fill that in here so that we can save that information
                    # ACCOUNT FOR BJ
                    if (
                        not self.player_action_list[0]
                        and not self.player_action_list[1]
                    ):
                        self.player_hand1 = [
                            self.player_card_list[0],
                            self.player_card_list[1],
                        ]
                        h1_action = self.BasicStrategy.get_action(
                            self.dealer_cards[0],
                            self.player_hand1,
                        )
                        self.player_action_list[0] = True
                        self.player_action_list[1] = True
                        self.player_hand1_total = self.BasicStrategy.total
                        self.player_hand1_done = True

                    elif not self.player_hand1_done:
                        index = self.player_action_list.index(False)
                        self.player_hand1.append(self.player_card_list[index])
                        h1_action = self.BasicStrategy.get_action(
                            self.dealer_cards[0],
                            self.player_hand1,
                        )
                        self.player_action_list[index] = True
                        self.player_hand1_total = self.BasicStrategy.total
                        self.player_hand1_done = True

                    if not self.player_hand2_done and self.splitting:
                        index = self.player_action_list.index(False)
                        self.player_hand2.append(self.player_card_list[index])
                        h2_action = self.BasicStrategy.get_action(
                            self.dealer_cards[0],
                            self.player_hand2,
                        )
                        self.player_action_list[index] = True
                        self.player_hand2_total = self.BasicStrategy.total
                        self.player_hand2_done = True

                    # print("Hand1 ", self.player_hand1)
                    # print("Hand1 Total:", self.player_hand1_total)

                    self.outcome = self.game_eval(
                        self.DealerCards.total,
                        self.player_hand1_total,
                        len(self.player_hand1),
                    )
                    if self.outcome == "WIN":
                        self.bankroll = self.bankroll + self.current_bet_1
                    elif self.outcome == "LOSE":
                        self.bankroll = self.bankroll - self.current_bet_1
                    elif self.outcome == "BLACKJACK":
                        self.bankroll = self.bankroll + (
                            self.current_bet_1 + round(self.current_bet_1 / 2, 2)
                        )
                    # print("Outcome:", outcome)
                    if self.splitting:
                        # RECORD SECOND HAND
                        # print("Hand2:", self.player_hand2)
                        # print("Hand2 Total:", self.player_hand2_total)
                        self.outcome2 = self.game_eval(
                            self.DealerCards.total,
                            self.player_hand2_total,
                            len(self.player_hand2),
                        )
                        if self.outcome == "WIN":
                            self.bankroll = self.bankroll + self.current_bet_2
                        elif self.outcome == "LOSE":
                            self.bankroll = self.bankroll - self.current_bet_2
                        elif self.outcome == "BLACKJACK":
                            self.bankroll = self.bankroll + (
                                self.current_bet_2 + round(self.current_bet_2 / 2, 2)
                            )
                    #     print("Outcome:", outcome2)

                    # print("Dealer Cards:", self.dealer_cards)
                    # print("Dealer Total:", self.DealerCards.total)

                    self.splitting = False
                    self.player_hand1_done = False
                    self.player_hand2_done = False
                    self.insurance_action = False
                    self.insurance = False

                if hand_done:
                    # When hands done uncheck play_next_hand and check is playing
                    self.player_action_list = []
                    # SAVE DATA TO DATA FILE
                    self.hand_cnt = self.hand_cnt + 1
                    format_dealer_cards = " ".join(self.dealer_cards)
                    format_player_cards = " ".join(self.player_card_list)
                    format_hand_1 = " ".join(self.player_hand1)
                    format_hand_2 = " ".join(self.player_hand2)
                    format_deck_length = str(len(self.deck))
                    format_current_bet_1 = str(self.current_bet_1)
                    format_current_bet_2 = str(self.current_bet_2)
                    format_true_cnt = str(self.wong_true_cnt)
                    format_bankroll = str(self.bankroll)
                    format_total_hand_1 = str(self.player_hand1_total)
                    format_total_hand_2 = str(self.player_hand2_total)
                    current_datetime = datetime.now()
                    current_date_time = current_datetime.strftime("%m/%d/%Y, %H:%M:%S")
                    # Dealer Cards
                    # Player cards
                    # Deck Length
                    # Bet Amount
                    # True Count
                    # Hand 1
                    # Hand 1 Res
                    # Hand 2
                    # Hand 2 Res
                    string = f"{format_bankroll}  [{format_dealer_cards}]  [{format_player_cards}]  {format_deck_length}  {format_true_cnt}  [{format_hand_1}]  {format_total_hand_1}  {format_current_bet_1}  {self.outcome}  [{format_hand_2}]  {format_total_hand_2}  {format_current_bet_2}  {self.outcome2}  {current_date_time}  {self.hand_cnt}\n"
                    file1 = open("game_data.txt", "a")
                    file1.write(string)
                    file1.close()

                    self.current_bet_1 = 0
                    self.current_bet_2 = 0
                    self.outcome = ""
                    self.outcome2 = ""
                    self.player_hand1 = []
                    self.player_hand2 = []
                    self.wanna_play()

                    if self.play_next_hand:
                        self.play_next_hand = False
                        self.is_playing = True
                # NOTE HANDLE PLAY FUNCTION END
                self.wanna_play()
                print(
                    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nDealer Cards:",
                    " ".join(self.dealer_cards),
                )
                print("Player Cards:", " ".join(self.player_card_list))
                print("Deck Length", len(self.deck))
                print("Last Removed", self.last_removed)
                print("current_card_list", self.CardSlot.current_card_list)
                print("dealer_current_card", self.DealerCards.current_card)
                # print("True Count:", self.thorp_true_cnt)
                # print("Thorpe Running", self.thorp_running_cnt)
                print("Wong True", self.wong_true_cnt)
                print("Wong Running", self.wong_running_cnt)
                print("Time-Left", self.time_left)
                print("Dealer_Cards_Total", self.DealerCards.total)
                print("Dealer Pixels", dealer_card_pixels)
                print("Dealer_Bools", self.DealerCards.card_bool_list)
                print("Pixel Precedent", self.DealerCards.pixel_precedent)
                print("Count", self.frame_count)
                print("Hand1:", self.player_hand1)
                print("Hand1 Done:", self.player_hand1_done)
                print("Hand2:", self.player_hand2)
                print("Hand2 Done:", self.player_hand2_done)
                print("Last Action:", self.BasicStrategy.action)
                print("Clock On:", self.clock_on)
                print("Hand Done:", self.hand_done)
                print("Game Gap:", self.game_gap)
                print("Can Place Bet:", self.can_place_bet)
                print("Bet Placed", self.bet_placed)
                print("Betting Amount:", self.betting_amount)
                print("Do Action:", self.can_do_action)
                print("play_next_hand:", self.play_next_hand)
                print("is_playing", self.is_playing)
                print("Shot Clock Px:", self.shot_clock_px)
                print("Shoe px:", self.ShoeWatcher.shoe_px)
                print("Shoe_ended:", self.CardSlot.shoe_ended)
                print("SHOE DONE", shoe_done)
                print("cut_card_px:", self.CardSlot.cut_card_px)
                # print("card_slot_px", self.CardSlot.card_slot_px)
                print("Dealer Stoppage", self.dealer_stoppage)
                print("can_new_card:", self.can_new_card)
                print("Facedown_card", facedown_card)
                print("Dealer_Start", dealer_start)
                print("Processed Hand", self.hand_proccessed)
                # print("new_card_px", self.NewCardWatcher.new_card_watcher_px)
                # print("motion_px", self.CardSlot.motion_px)

                self.prev_player_card_len = self.player_card_len

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            self.frame_count = self.frame_count + 1
            count = count + 1


# game = Game("../../../Desktop/6_card_+2.mp4")
# game = Game("../../../Desktop/test_play1.mp4")
# game = Game("../../../Desktop/split_aces.mp4")
# game = Game("../../../Desktop/split_8s.mp4")
# game = Game("../../../Desktop/insurance.mp4")
game = Game(58)
game.main()
