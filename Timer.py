import time
import math


class Timer:
    """Have the ablility to set a time limit start and check if time reached returning bool"""

    def __init__(self, minutes=0, seconds=0):
        self.start_time = 0
        self.time_limit = minutes * 60 + seconds
        self.timer_on = False
        self.timer_done = False
        self.time_left = 0

    def init_timer(self):
        """Start the timer"""
        self.timer_on = True
        self.timer_done = False
        self.start_time = round(math.ceil(time.time() * 100) / 100)
        # print("timer_start")

    def compare_time(self):
        """Check the current time against the time we start to see if we have reached out time limit (minutes and seconds)"""
        # if self.start_time == 0:
        #     self.init_timer()

        current_time = round(math.ceil(time.time() * 100) / 100)
        timepast = current_time - self.start_time
        self.time_left = self.time_limit - timepast
        formatted_time = self.time_formatter(timepast)
        message = f"Time elapsed {formatted_time}"

        if timepast >= self.time_limit and self.start_time > 0:
            self.timer_done = True
            return (True, message)
        else:
            return (False, message)

    def time_formatter(self, time_in_seconds):
        """Format time from seconds"""
        minutes = math.floor(time_in_seconds / 60)
        seconds = time_in_seconds - (minutes * 60)

        return f"{minutes}:{seconds}"


if __name__ == "__main__":
    timer = Timer(0, 5)
    timer.init_timer()
    time.sleep(10)
    boolean, message = timer.compare_time()
    print(boolean, message)
