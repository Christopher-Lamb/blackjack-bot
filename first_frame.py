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


main_dict = {
    "game_width": 1442,
    "game_height": 871,
    "top_left": (209, 113),
    "bottom_right": (1651, 984),
    "card_slot": [(694, 555), (748, 599)],
    "shoe_deck": [(1018, 490), (1063, 517)],
    "player_action": [(1262, 588), (1379, 642)],
    "dealer_cards": [(433, 468), (577, 490)],
    "dollar_chip": (595, 416),
    "double_btn": (554, 422),
    "hit_btn": (665, 422),
    "stand_btn": (775, 422),
    "split_btn": (883, 422),
    "one_card": [(439, 468), (479, 490)],
    "one_card_active": (445, 493),
    "two_card": [(451, 468), (490, 490)],
    "two_card_active": (476, 493),
    "three_card": [(470, 468), (514, 490)],
    "three_card_active": (493, 493),
    "four_card": [(486, 468), (526, 490)],
    "four_card_active": (507, 493),
    "five_card": [(501, 468), (542, 490)],
    "five_card_active": (529, 490),
    "dealer_white": [(435, 490), (533, 498)],
    "make_bet_btn": (719, 721),
    "shot_clock": [(744, 289), (762, 302)],
}


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


def get_frame(frame, str, dict, px=0):
    """Get frame from coords"""
    img = {}
    try:
        img = frame[
            dict[str][0][1] - px : dict[str][1][1] + px,
            dict[str][0][0] - px : dict[str][1][0] + px,
        ]
    except TypeError:
        pass

    return img


def get_boxes(img, top_left, bottom_right):
    """Take the dimensions of our window/game and return the parts of blackjack we want to look at"""
    game_width = bottom_right[0] - top_left[0]
    game_height = bottom_right[1] - top_left[1]

    x, y = top_left
    x2, y2 = bottom_right
    card_box_width = 30

    main_dict = {
        "game_width": 1442,
        "game_height": 871,
        "top_left": (x, y),
        "bottom_right": (x2, y2),
        "game": [(x, y), (x2, y2)],
        "card_slot": [(694 + x, y + 555), (748 + x, y + 599)],
        "shoe_deck": [(1018 + x, y + 490), (1063 + x, y + 517)],
        "player_action": [(1262 + x, y + 588), (1379 + x, y + 642)],
        "dealer_cards": [(433 + x, y + 468), (577 + x, y + 490)],
        "dollar_chip": (595 + x, y + 416),
        "double_btn": (554 + x, y + 422),
        "hit_btn": (665 + x, y + 422),
        "stand_btn": (775 + x, y + 422),
        "split_btn": (883 + x, y + 422),
        "one_card": [(439 + x, y + 472), (439 + x + card_box_width, y + 490)],
        "two_card": [(455 + x, y + 472), (455 + x + card_box_width, y + 490)],
        "three_card": [(471 + x, y + 472), (471 + x + card_box_width, y + 490)],
        "four_card": [(487 + x, y + 472), (487 + x + card_box_width, y + 490)],
        "five_card": [(503 + x, y + 472), (503 + x + card_box_width, y + 490)],
        "one_card_active": (445 + x, y + 493),
        "two_card_active": (476 + x, y + 493),
        "three_card_active": (493 + x, y + 493),
        "four_card_active": (507 + x, y + 493),
        "five_card_active": (529 + x, y + 490),
        "make_bet_btn": (719 + x, y + 721),
        "shot_clock": [(744 + x, y + 289), (762 + x, y + 302)],
        "new_card_watcher": [(995 + x, y + 520), (995 + 20 + x, y + 520 + 20)],
    }
    dict = {
        "game_width": 1442,
        "game_height": 871,
        "game": (0, 255, 0),
        "top_left": (255, 255, 255),
        "bottom_right": (255, 255, 255),
        "card_slot": (35, 99, 186),
        "shoe_deck": (207, 27, 197),
        "player_action": (0, 0, 244),
        "dealer_cards": (241, 126, 65),
        "dollar_chip": (0, 215, 255),
        "double_btn": (49, 77, 255),
        "hit_btn": (115, 244, 0),
        "stand_btn": (20, 24, 255),
        "split_btn": (228, 113, 0),
        "one_card": (0, 255, 0),
        "one_card_active": (255, 255, 255),
        "two_card": (0, 255, 255),
        "two_card_active": (255, 255, 255),
        "three_card": (0, 0, 255),
        "three_card_active": (255, 255, 255),
        "four_card": (255, 0, 255),
        "four_card_active": (255, 255, 255),
        "five_card": (255, 0, 0),
        "five_card_active": (255, 255, 255),
        "make_bet_btn": (0, 215, 155),
        "shot_clock": (81, 238, 100),
        "new_card_watcher": (255, 255, 0),
    }

    # PRINT BOXES
    for k, v in main_dict.items():
        if k in [
            "one_card",
            "two_card",
            "three_card",
            "four_card",
            "five_card",
            "dealer_cards",
            "one_card_active",
            "two_card_active",
            "three_card_active",
            "four_card_active",
            "five_card_active",
        ]:
            continue
        if isinstance(v, list):
            if k == "game":
                img = cv2.rectangle(img, v[0], v[1], dict[k], 5)
            else:
                img = cv2.rectangle(img, v[0], v[1], dict[k], 1)
        if isinstance(v, tuple):
            if k in [
                "hit_btn",
                "stand_btn",
                "make_bet_btn",
                "split_btn",
                "double_btn",
                "dollar_chip",
            ]:
                img = cv2.line(img, v, v, (0, 0, 0), 12)
                img = cv2.line(img, v, v, dict[k], 6)
            else:
                img = cv2.line(img, v, v, (0, 0, 0), 4)
                img = cv2.line(img, v, v, dict[k], 1)

    return main_dict


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
        if count > 0:
            # print("first_frame")
            try:
                dict = get(frame)
                coord = dict["top_left"]

                frame = cv2.line(frame, coord, coord, (255, 255, 0), 4)
                frame[coord] = [255, 255, 0]
                # cv2.imwrite("blackjack_frame_guide.jpg", frame)
                gotten_frame = get_frame(frame, "one_card", dict)
                gotten_frame2 = get_frame(frame, "two_card", dict)
                gotten_frame3 = get_frame(frame, "three_card", dict)
                gotten_frame4 = get_frame(frame, "four_card", dict)
                gotten_frame5 = get_frame(frame, "five_card", dict)
                # print(dict)
                cv2.imshow("FRAME", frame)
                gotten_frame = cv2.resize(gotten_frame, (0, 0), fx=8, fy=8)
                gotten_frame2 = cv2.resize(gotten_frame2, (0, 0), fx=8, fy=8)
                gotten_frame3 = cv2.resize(gotten_frame3, (0, 0), fx=8, fy=8)
                gotten_frame4 = cv2.resize(gotten_frame4, (0, 0), fx=8, fy=8)
                gotten_frame5 = cv2.resize(gotten_frame5, (0, 0), fx=8, fy=8)
                cv2.imshow("Gotten Frame", gotten_frame)
                cv2.imshow("Gotten Frame2", gotten_frame2)
                cv2.imshow("Gotten Frame3", gotten_frame3)
                cv2.imshow("Gotten Frame4", gotten_frame4)
                cv2.imshow("Gotten Frame5", gotten_frame5)
            except:
                continue
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
