import pyautogui as pygui
import numpy as np
import random
import time


sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)


class BlackjackGUI:
    screen = 1

    def __init__(self):
        """Init lol"""
        self.hit_btn = (0, 0)
        self.stand_btn = (0, 0)
        self.double_btn = (0, 0)
        self.split_btn = (0, 0)
        self.dollar_chip = (0, 0)
        self.make_bet_btn = (0, 0)
        self.destination = (0, 0)
        self.points = []

    def build_coords(
        self, dict, hit_btn, stand_btn, double_btn, split_btn, dollar_chip, make_bet_btn
    ):
        """Build coords"""
        num = 1920 * (self.screen - 1)
        self.hit_btn = (dict[hit_btn][0] + num, dict[hit_btn][1])
        self.stand_btn = (dict[stand_btn][0] + num, dict[stand_btn][1])
        self.double_btn = (dict[double_btn][0] + num, dict[double_btn][1])
        self.split_btn = (dict[split_btn][0] + num, dict[split_btn][1])
        self.dollar_chip = (dict[dollar_chip][0] + num, dict[dollar_chip][1])
        self.make_bet_btn = (dict[make_bet_btn][0] + num, dict[make_bet_btn][1])

    def get_idle_coords(self):
        """Return random off screen coords"""
        num = 1920 * (self.screen - 1)
        random_coords = (
            random.randint(150 + num, 1800 + num),
            random.randint(950, 1080),
        )
        return random_coords

    def premove(self):
        """Move so that you dont interfere with other things"""
        num = 1920 * (self.screen - 1)
        pygui.moveTo(
            540 + num,
            1080,
            _pause=False,
        )

    def preclick(self):
        """Click on screen below betting btn to actiavte screen interactivity"""
        x, y = self.dollar_chip
        new_x = random.randint(x - 50, x + 50)
        new_y = random.randint(y + 100, y + 200)
        self.move_mouse(pygui.position(), (new_x, new_y))
        pygui.click()

    def player_action(self, action="STAND"):
        """Carry out location and click once"""
        # currentMouseX, currentMouseY = pygui.position()
        # print(currentMouseX,currentMouseY)
        coords = (0, 0)
        if action == "STAND":
            coords = self.stand_btn
        elif action == "HIT":
            coords = self.hit_btn
        elif action == "DOUBLE":
            coords = self.double_btn
        elif action == "SPLIT":
            coords = self.split_btn

        x, y = coords
        rand_x = x + random.randint(-5, 5)
        rand_y = y + random.randint(-5, 5)
        coords = (rand_x, rand_y)
        sec_sleep1 = random.randint(70, 110) / 100
        sec_sleep2 = random.randint(50, 210) / 100

        out_coords = self.get_idle_coords()
        self.premove()
        self.move_mouse(pygui.position(), coords)
        time.sleep(sec_sleep1)
        pygui.click()
        pygui.click()
        time.sleep(sec_sleep2)
        self.move_mouse(pygui.position(), out_coords)

    def place_bet(self, amount=1):
        """Select dollar amount then click make bet that amount of times"""

        out_coords = self.get_idle_coords()
        x1, y1 = self.dollar_chip
        x2, y2 = self.make_bet_btn
        rand_x1 = x1 + random.randint(-5, 5)
        rand_y1 = y1 + random.randint(-5, 5)
        rand_x2 = x2 + random.randint(-5, 5)
        rand_y2 = y2 + random.randint(-5, 5)
        rand_interval = random.randint(10, 22) / 100
        self.premove()
        self.preclick()
        self.move_mouse(pygui.position(), (rand_x1, rand_y1))
        pygui.click()
        self.move_mouse(pygui.position(), (rand_x2, rand_y2))
        pygui.click(
            clicks=amount,
            interval=rand_interval,
        )
        self.move_mouse(pygui.position(), out_coords)

    def wind_mouse(
        self,
        start_x,
        start_y,
        dest_x,
        dest_y,
        G_0=10,
        W_0=10,
        M_0=25,
        D_0=20,
        move_mouse=lambda x, y: None,
    ):
        """
        WindMouse algorithm. Calls the move_mouse kwarg with each new step.
        Released under the terms of the GPLv3 license.
        G_0 - magnitude of the gravitational force
        W_0 - magnitude of the wind force fluctuations
        M_0 - maximum step size (velocity clip threshold)
        D_0 - distance where wind behavior changes from random to damped
        """
        current_x, current_y = start_x, start_y
        v_x = v_y = W_x = W_y = 0
        while (dist := np.hypot(dest_x - start_x, dest_y - start_y)) >= 1:
            W_mag = min(W_0, dist)
            if dist >= D_0:
                W_x = W_x / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
                W_y = W_y / sqrt3 + (2 * np.random.random() - 1) * W_mag / sqrt5
            else:
                W_x /= sqrt3
                W_y /= sqrt3
                if M_0 < 3:
                    M_0 = np.random.random() * 3 + 3
                else:
                    M_0 /= sqrt5
            v_x += W_x + G_0 * (dest_x - start_x) / dist
            v_y += W_y + G_0 * (dest_y - start_y) / dist
            v_mag = np.hypot(v_x, v_y)
            if v_mag > M_0:
                v_clip = M_0 / 2 + np.random.random() * M_0 / 2
                v_x = (v_x / v_mag) * v_clip
                v_y = (v_y / v_mag) * v_clip
            start_x += v_x
            start_y += v_y
            move_x = int(np.round(start_x))
            move_y = int(np.round(start_y))
            if current_x != move_x or current_y != move_y:
                # This should wait for the mouse polling interval
                move_mouse(current_x := move_x, current_y := move_y)

        return current_x, current_y

    def move_mouse(self, start_coords, end_coords):
        x1, y1 = start_coords
        x2, y2 = end_coords

        points = []
        self.wind_mouse(x1, y1, x2, y2, move_mouse=lambda x, y: points.append([x, y]))
        count = 0
        while True:
            if count >= len(points) - 1:
                break
            count = count + 1
            x, y = points[count]
            pygui.moveTo(
                x,
                y,
                _pause=False,
            )


if __name__ == "__main__":
    gui = BlackjackGUI()
    num = 1920 * 2
    # x1, y1 = (860, 525)
    x1, y1 = (860 + num, 1080)
    x2, y2 = (860 + num, 625)

    points = []
    gui.wind_mouse(x1, y1, x2, y2, move_mouse=lambda x, y: points.append([x, y]))
    points = np.asarray(points)
    count = 0

    # print(len(points))
    while True:
        if count >= len(points) - 1:
            break
        count = count + 1
        x, y = points[count]
        pygui.moveTo(
            x,
            y,
            _pause=False,
        )
    # while True:
    #     x, y = pygui.position()
    #     num = 1920 * (gui.screen - 1)
    #     count = count + 1
    #     # print(x, y)
    #     if count == 20:
    #         break
