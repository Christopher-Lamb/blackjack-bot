import numpy as np
import cv2
import json
import random


def hex_to_rgb(hex):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal)

    return tuple(rgb)


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# img = cv2.imread("./assets/actions_blue.png")
# img_grid = img.copy()
# Template match the card slot
# print(card_box_template.shape)

# small_img = cv2.resize(small_img,(0,0),fx=.5,fy=.5)

# for i in range(img_grid.shape[0]):
# 	for j in range(img_grid.shape[1]):
#           b,g,r = img_grid[i][j]
#           if rgb_to_hex(r,g,b) in data:
#             img_grid[i][j] = [33,22,11]
#             # img_grid[i][j] = [100,225,225]


# h, w = card_box_template.shape
# look for rectangles and get the one close to the center

f = open("/Users/illmonk/Documents/python_scripts/blackjack-bot/yolov5/background.json")
data = json.load(f)


def plot_box(img, pt1, pt2):
    """Take points and plot them"""


def negate_colors(img):
    color_list = []
    img_grid = img.copy()
    thickness = 20
    left, right, top, bottom = (232, 1670, 116, 984)
    for x in range(left - thickness, left):
        for y in range(top - thickness, bottom + thickness):
            b, g, r = img_grid[y, x]
            color_list.append(rgb_to_hex(r, g, b))
            img_grid[y, x] = [0, 50, 0]
    for x in range(left, right):
        for y in range(bottom, bottom + thickness):
            b, g, r = img_grid[y, x]
            color_list.append(rgb_to_hex(r, g, b))
            img_grid[y, x] = [0, 50, 0]

    for x in range(right, right + thickness):
        for y in range(top - thickness, bottom + thickness):
            b, g, r = img_grid[y, x]
            color_list.append(rgb_to_hex(r, g, b))
            img_grid[y, x] = [0, 50, 0]

    for x in range(left, right):
        for y in range(top - thickness, top):
            b, g, r = img_grid[y, x]
            color_list.append(rgb_to_hex(r, g, b))
            img_grid[y, x] = [0, 50, 0]

    color_list = list(set(color_list))

    return (color_list, img_grid)


def get_corners(img):
    height, width = img.shape
    top_left = []
    bottom_right = []
    location = (width // 2, height // 2)
    half_w, half_h = location
    for x in range(0, half_w):
        val = img[half_h, half_w - x]
        img[half_h, half_w - x] = 200
        if val == 255:
            # print(half_h, half_w - x)
            top_left.append(half_w - x)
            break
    for y in range(0, half_h):
        val = img[half_h - y, half_w]
        img[half_h - y, half_w] = 200
        if val == 255:
            # print(half_h - y, half_w)
            top_left.append(half_h - y)
            break

    for x in range(0, half_w):
        val = img[half_h, half_w + x]
        img[half_h, half_w + x] = 200
        if val == 255:
            # print(half_h, half_w + x)
            bottom_right.append(half_w + x)
            break
    for y in range(0, half_h):
        val = img[half_h + y, half_w]
        img[half_h + y, half_w] = 200
        if val == 255:
            # print(half_h + y, half_w)
            bottom_right.append(half_h + y)
            break
    img = cv2.line(img, (location), (location), (255, 255, 255), 15)
    # print("tl br:", top_left, bottom_right)
    return (top_left, bottom_right)


def get_boxes(img, top_left, bottom_right):
    """Take the dimensions of our window/game and return the parts of blackjack we want to look at"""
    game_width = bottom_right[0] - top_left[0]
    game_height = bottom_right[1] - top_left[1]

    card_slot_top_left = (
        (round(game_width * (77 / 160))) + top_left[0],
        (round(game_height * (51 / 80))) + top_left[1],
    )
    card_slot_bottom_right = (
        (round(game_width * (83 / 160))) + top_left[0],
        (round(game_height * (55 / 80))) + top_left[1],
    )

    # print(game_width, game_height)
    # cv2.line(img, (card_slot_top_left[0], card_slot_top_left[1]),
    #  (card_slot_top_left[0], card_slot_top_left[1]), (0, 255, 0), 10)
    # cv2.line(img, (card_slot_bottom_right[0], card_slot_bottom_right[1]), (
    # card_slot_bottom_right[0], card_slot_bottom_right[1]), (255, 100, 0), 10)
    cv2.rectangle(img, card_slot_top_left, card_slot_bottom_right, (0, 0, 255), 4)

    player_action_top_left = (
        (round(game_width * (35 / 40))) + top_left[0],
        (round(game_height * (27 / 40))) + top_left[1],
    )
    player_action_bottom_right = (
        (round(game_width * (153 / 160))) + top_left[0],
        (round(game_height * (59 / 80))) + top_left[1],
    )
    cv2.line(
        img,
        (player_action_top_left[0], player_action_top_left[1]),
        (player_action_top_left[0], player_action_top_left[1]),
        (255, 0, 0),
        10,
    )
    cv2.line(
        img,
        (player_action_bottom_right[0], player_action_bottom_right[1]),
        (player_action_bottom_right[0], player_action_bottom_right[1]),
        (0, 255, 0),
        10,
    )
    cv2.rectangle(
        img, player_action_top_left, player_action_bottom_right, (0, 0, 255), 4
    )

    dealer_cards_top_left = (
        (round(game_width * (24 / 80))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    dealer_cards_bottom_right = (
        (round(game_width * (32 / 80))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )
    # cv2.line(
    #     img,
    #     (dealer_cards_top_left[0], dealer_cards_top_left[1]),
    #     (dealer_cards_top_left[0], dealer_cards_top_left[1]),
    #     (200, 200, 200),
    #     4,
    # )
    # cv2.line(
    #     img,
    #     (dealer_cards_bottom_right[0], dealer_cards_bottom_right[1]),
    #     (dealer_cards_bottom_right[0], dealer_cards_bottom_right[1]),
    #     (0, 240, 240),
    #     4,
    # )
    # cv2.rectangle(
    #     img, dealer_cards_top_left, dealer_cards_bottom_right, (0, 255, 255), 4
    # )

    dollar_chip = (
        (round(game_width * (33 / 80))) + top_left[0],
        (round(game_height * (153 / 320))) + top_left[1],
    )
    cv2.line(
        img,
        (dollar_chip[0], dollar_chip[1]),
        (dollar_chip[0], dollar_chip[1]),
        (200, 200, 0),
        10,
    )
    # cv2.circle(img, (dollar_chip[0],dollar_chip[1]), 29 , (255,0,0),4)

    double_btn = (
        (round(game_width * (123 / 320))) + top_left[0],
        (round(game_height * (155 / 320))) + top_left[1],
    )
    cv2.line(
        img,
        (double_btn[0], double_btn[1]),
        (double_btn[0], double_btn[1]),
        (70, 92, 246),
        10,
    )
    cv2.circle(img, (double_btn[0], double_btn[1]), 5, (0, 0, 0), 2)

    hit_btn = (
        (round(game_width * (295 / 640))) + top_left[0],
        (round(game_height * (155 / 320))) + top_left[1],
    )
    cv2.line(
        img, (hit_btn[0], hit_btn[1]), (hit_btn[0], hit_btn[1]), (123, 192, 58), 10
    )
    cv2.circle(img, (hit_btn[0], hit_btn[1]), 5, (0, 0, 0), 2)

    stand_btn = (
        (round(game_width * (275 / 512))) + top_left[0],
        (round(game_height * (155 / 320))) + top_left[1],
    )
    cv2.line(
        img,
        (stand_btn[0], stand_btn[1]),
        (stand_btn[0], stand_btn[1]),
        (45, 26, 245),
        10,
    )
    cv2.circle(img, (stand_btn[0], stand_btn[1]), 5, (0, 0, 0), 2)

    split_btn = (
        (round(game_width * (49 / 80))) + top_left[0],
        (round(game_height * (155 / 320))) + top_left[1],
    )
    cv2.line(
        img,
        (split_btn[0], split_btn[1]),
        (split_btn[0], split_btn[1]),
        (0, 246, 255),
        10,
    )
    cv2.circle(img, (split_btn[0], split_btn[1]), 5, (0, 0, 0), 2)

    # ===================
    # ONE CARD
    # ===================
    one_card_top_left = (
        (round(game_width * (195 / 640))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    one_card_bottom_right = (
        (round(game_width * (425 / 1280))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )

    one_card_active = (
        (round(game_width * (395 / 1280))) + top_left[0],
        (round(game_height * (181 / 320))) + top_left[1],
    )
    # ===================
    # TWO CARD
    # ===================
    two_card_top_left = (
        (round(game_width * (400 / 1280))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    two_card_bottom_right = (
        (round(game_width * (435 / 1280))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )

    two_card_active = (
        (round(game_width * (845 / 2560))) + top_left[0],
        (round(game_height * (181 / 320))) + top_left[1],
    )
    # ===================
    # THREE CARD
    # ===================
    three_card_top_left = (
        (round(game_width * (417 / 1280))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    three_card_bottom_right = (
        (round(game_width * (456 / 1280))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )
    three_card_active = (
        (round(game_width * (438 / 1280))) + top_left[0],
        (round(game_height * (181 / 320))) + top_left[1],
    )
    # ===================
    # FOUR CARD
    # ===================
    four_card_top_left = (
        (round(game_width * (431 / 1280))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    four_card_bottom_right = (
        (round(game_width * (467 / 1280))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )
    four_card_active = (
        (round(game_width * (450 / 1280))) + top_left[0],
        (round(game_height * (181 / 320))) + top_left[1],
    )
    # ===================
    # FIVE CARD
    # ===================
    five_card_top_left = (
        (round(game_width * (445 / 1280))) + top_left[0],
        (round(game_height * (86 / 160))) + top_left[1],
    )
    five_card_bottom_right = (
        (round(game_width * (481 / 1280))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )
    five_card_active = (
        (round(game_width * (470 / 1280))) + top_left[0],
        (round(game_height * (360 / 640))) + top_left[1],
    )

    dealer_white_top_left = (
        (round(game_width * (193 / 640))) + top_left[0],
        (round(game_height * (180 / 320))) + top_left[1],
    )
    dealer_white_bottom_right = (
        (round(game_width * (473 / 1280))) + top_left[0],
        (round(game_height * (183 / 320))) + top_left[1],
    )
    cv2.rectangle(
        img, dealer_white_top_left, dealer_white_bottom_right, (0, 255, 255), 1
    )

    shoe_deck_top_left = (
        (round(game_width * (113 / 160))) + top_left[0],
        (round(game_height * (90 / 160))) + top_left[1],
    )
    shoe_deck_bottom_right = (
        (round(game_width * (118 / 160))) + top_left[0],
        (round(game_height * (95 / 160))) + top_left[1],
    )
    cv2.line(
        img,
        (shoe_deck_top_left[0], shoe_deck_top_left[1]),
        (shoe_deck_top_left[0], shoe_deck_top_left[1]),
        (200, 200, 0),
        10,
    )
    cv2.line(
        img,
        (shoe_deck_bottom_right[0], shoe_deck_bottom_right[1]),
        (shoe_deck_bottom_right[0], shoe_deck_bottom_right[1]),
        (0, 240, 240),
        10,
    )
    cv2.rectangle(img, shoe_deck_top_left, shoe_deck_bottom_right, (0, 255, 255), 4)

    shot_clock_top_left = (
        (round(game_width * (330 / 640))) + top_left[0],
        (round(game_height * (106 / 320))) + top_left[1],
    )
    shot_clock_bottom_right = (
        (round(game_width * (338 / 640))) + top_left[0],
        (round(game_height * (111 / 320))) + top_left[1],
    )

    make_bet_btn = (
        (round(game_width * (319 / 640))) + top_left[0],
        (round(game_height * (265 / 320))) + top_left[1],
    )

    dict = {
        "game_width": game_width,
        "game_height": game_height,
        "top_left": tuple(top_left),
        "bottom_right": tuple(bottom_right),
        "card_slot": [card_slot_top_left, card_slot_bottom_right],
        "shoe_deck": [shoe_deck_top_left, shoe_deck_bottom_right],
        "player_action": [player_action_top_left, player_action_bottom_right],
        "dealer_cards": [dealer_cards_top_left, dealer_cards_bottom_right],
        "dollar_chip": dollar_chip,
        "double_btn": double_btn,
        "hit_btn": hit_btn,
        "stand_btn": stand_btn,
        "split_btn": split_btn,
        "one_card": [one_card_top_left, one_card_bottom_right],
        "one_card_active": one_card_active,
        "two_card": [two_card_top_left, two_card_bottom_right],
        "two_card_active": two_card_active,
        "three_card": [three_card_top_left, three_card_bottom_right],
        "three_card_active": three_card_active,
        "four_card": [four_card_top_left, four_card_bottom_right],
        "four_card_active": four_card_active,
        "five_card": [five_card_top_left, five_card_bottom_right],
        "five_card_active": five_card_active,
        "dealer_white": [dealer_white_top_left, dealer_white_bottom_right],
        "make_bet_btn": make_bet_btn,
        "shot_clock": [shot_clock_top_left, shot_clock_bottom_right],
    }

    return dict


def dot(frame, coord_list):
    for coord in coord_list:
        frame = cv2.line(frame, coord, coord, (0, 0, 255), 1)


def coords_to_pixels(position_dict, frame, coords):
    """Take a list of coords turn into pixels"""
    pixels = []
    for coord in coords:
        color = frame[
            position_dict[coord][1] : position_dict[coord][1] + 1,
            position_dict[coord][0] : position_dict[coord][0] + 1,
        ]
        # color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pixels.append(color[0][0])
    return pixels


def get(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    upper_blue = np.array([140, 255, 255])
    lower_blue = np.array([90, 210, 150])

    # print(frame[4:8, 10:14])
    # frame[4:8, 10:14] = [0, 0, 0]

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    top_left, bottom_right = get_corners(mask)

    # cv2.imshow("MASK", mask)
    # print("tl bl",top_left, bottom_right)
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.rectangle(img, (top_left),
    #               (bottom_right), (0, 255, 0), 4)
    # img = cv2.putText(
    #     img, 'Game', (top_left[0], top_left[1]-10), font, .5, (0, 255, 0), 1, cv2.LINE_AA)

    dict = get_boxes(img, top_left, bottom_right)
    return dict


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("../../../Desktop/dealer_cards_test2.mp4")
    count = 0
    dict = {}
    while cap.isOpened():
        """Main Game Video Loop"""
        ret, frame = cap.read()
        # blank = frame.copy()
        # blank = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        if count == 0:
            # print("first_frame")
            dict = get(frame)
            
              
                

            print(dict)
            cv2.imshow("FRAME",frame)

        elif (count % 1) == 0:
            """"""
            # frame = cv2.line(
            #     frame, dict["five_card_active"], dict["five_card_active"], (255, 255, 255), 4
            # )
            # print(frame[dict["one_card_active"][1], dict["one_card_active"][0]])
            # frame[dict["make_bet_btn"][1], dict["make_bet_btn"][0]] = [
            #     255,
            #     255,
            #     255,
            # ]
            # shoe_frame = frame[
            #     dict["shot_clock"][0][1] : dict["shot_clock"][1][1],
            #     dict["shot_clock"][0][0] : dict["shot_clock"][1][0],
            # ]
            # print(shoe_frame)

            # shoe_frame = cv2.resize(shoe_frame, (0, 0), fx=4, fy=4)
            # cv2.imshow("Shoe Frame", shoe_frame)
            # cv2.imshow("Frame", frame)
            # color_list, new_img = negate_colors(frame)
            # cv2.imshow("Frame", new_img)
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # upper_blue = np.array([140, 255, 255])
            # lower_blue = np.array([90, 210, 150])

            # # print(frame[4:8, 10:14])
            # # frame[4:8, 10:14] = [0, 0, 0]

            # mask = cv2.inRange(hsv, lower_blue, upper_blue)
            # number_of_white_pix = np.sum(mask == 255)
            # result = cv2.bitwise_and(frame, frame, mask=mask)
            # cv2.imshow("Frame", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count = count + 1


# main(img)

# try:
#     main(img)
# except IndexError:
#     ''''''
# green = np.uint8([[[55, 55, 55]]])

# # Convert Green color to Green HSV
# hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsv_green)
