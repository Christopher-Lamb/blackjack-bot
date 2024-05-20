class BasicStrategy:
    double_after_split = True

    def __init__(self):
        self.upcard = ""
        self.action = ""
        self.can_double = False
        self.can_split = False
        self.total = 0
        self.player_cards = []
        self.true_count = 0

    def handle_split(self):
        """Check all possible split actions"""
        # LOOK UP DOUBLE AFTER SPLIT
        card = self.player_cards[0]
        upcard = self.upcard
        if card in ["11", "8"]:
            self.action = "SPLIT"
        elif card == "5":
            self.action = "CONT"
        elif card == "10":
            if self.true_count >= 6 and upcard == "4":
                self.action = "SPLIT"
            elif self.true_count >= 5 and upcard == "5":
                self.action = "SPLIT"
            elif self.true_count >= 4 and upcard == "6":
                self.action = "SPLIT"
            else:
                self.action = "CONT"
        elif card == "9" and upcard in ["2", "3", "4", "5", "6", "8", "9"]:
            self.action = "SPLIT"

        elif card == "7" and upcard in ["2", "3", "4", "5", "6", "7"]:
            self.action = "SPLIT"

        elif card == "6" and upcard in ["2", "3", "4", "5", "6", "7"]:
            if upcard == "2" and not self.double_after_split:
                self.action = "CONT"
            else:
                self.action = "SPLIT"
        elif card == "4" and upcard in ["5", "6"] and self.double_after_split:
            self.action = "SPLIT"
        elif card in ["2", "3"] and upcard in ["2", "3", "4", "5", "6", "7"]:
            if upcard in ["2", "3"] and not self.double_after_split:
                self.action = "CONT"
            else:
                self.action = "SPLIT"
        else:
            self.action = "CONT"

    def handle_soft(self):
        """Check all possible split actions"""
        card = self.total - 11

        if card == 9:
            self.action = "STAND"
        elif card == 8:
            if self.upcard == "6" and self.can_double:
                if self.true_count < 0:
                    self.action = "STAND"
                else:
                    self.action = "DOUBLE"
            elif self.upcard == "5" and self.can_double and self.true_count >= 1:
                self.action = "DOUBLE"
            elif self.upcard == "4" and self.can_double and self.true_count >= 3:
                self.action = "DOUBLE"
            else:
                self.action = "STAND"

        elif card == 7:
            if self.upcard in ["2", "3", "4", "5", "6", "7", "8"]:
                if self.upcard in ["2", "3", "4", "5", "6"] and self.can_double:
                    self.action = "DOUBLE"
                else:
                    self.action = "STAND"
            else:
                self.action = "HIT"
        elif card == 6:
            if self.upcard in ["3", "4", "5", "6"] and self.can_double:
                self.action = "DOUBLE"
            else:
                if self.upcard == "2" and self.can_double and self.true_count >= 1:
                    self.action = "DOUBLE"
                else:
                    self.action = "HIT"
        elif card in [4, 5]:
            if self.upcard in ["4", "5", "6"] and self.can_double:
                self.action = "DOUBLE"
            else:
                self.action = "HIT"
        elif card in [2, 3]:
            if self.upcard in ["5", "6"] and self.can_double:
                self.action = "DOUBLE"
            else:
                self.action = "HIT"

    def handle_hard(self):
        """Check all possible split actions"""
        if self.total >= 17:
            self.action = "STAND"
        elif self.total in [16, 15, 14, 13]:
            if self.upcard in ["2", "3", "4", "5", "6"]:
                if self.total == 13 and self.upcard == "2" and self.true_count <= -1:
                    self.action = "HIT"
                else:
                    self.action = "STAND"
            elif self.total == 16:
                if self.upcard == "9" and self.true_count >= 4:
                    self.action = "STAND"
                elif self.upcard == "10" and self.true_count >= 0:
                    self.action = "STAND"
                elif self.upcard == "11" and self.true_count >= 3:
                    self.action = "STAND"
                else:
                    self.action = "HIT"
            elif self.total == 15:
                if self.upcard == "10" and self.true_count >= 4:
                    self.action = "STAND"
                elif self.upcard == "11" and self.true_count >= 5:
                    self.action = "STAND"
                else:
                    self.action = "HIT"
            # elif self.upcard =
            else:
                self.action = "HIT"
        elif self.total == 12:
            if self.upcard == "2" and self.true_count >= 3:
                self.action = "STAND"
            elif self.upcard == "3" and self.true_count >= 2:
                self.action = "STAND"
            elif self.upcard in ["4", "5", "6"]:
                if self.upcard == "4" and self.true_count <= 0:
                    self.action = "HIT"
                else:
                    self.action = "STAND"
            else:
                self.action = "HIT"
        elif self.total == 11:
            if self.can_double:
                self.action = "DOUBLE"
            else:
                self.action = "HIT"
        elif self.total == 10:
            if self.upcard in ["10", "11"]:
                if self.upcard == "10" and self.true_count >= 4 and self.can_double:
                    self.action = "DOUBLE"
                elif self.upcard == "11" and self.true_count >= 3 and self.can_double:
                    self.action = "DOUBLE"
                else:
                    self.action = "HIT"
            else:
                if self.can_double:
                    self.action = "DOUBLE"
                else:
                    self.action = "HIT"
        elif self.total == 9:
            if self.upcard == "2" and self.true_count >= 1 and self.can_double:
                self.action = "DOUBLE"
            elif self.upcard in ["3", "4", "5", "6"]:
                if self.can_double:
                    self.action = "DOUBLE"
                else:
                    self.action = "HIT"
            elif self.upcard == "7" and self.true_count >= 3 and self.can_double:
                self.action = "DOUBLE"
            else:
                self.action = "HIT"
        elif (
            self.total == 8
            and self.upcard == "6"
            and self.true_count >= 2
            and self.can_double
        ):
            self.action = "DOUBLE"
        else:
            self.action = "HIT"

    def get_total(self, cards=[]):
        """Take a list of player cards return total and if soft or hard"""
        total = 0
        is_soft = False
        ace_count = 0
        self.can_double = True
        player_cards = self.player_cards
        if len(cards) > 0:
            player_cards = cards

        # Handle can double and can split
        if len(player_cards) == 2:
            if int(player_cards[0]) == int(player_cards[1]):
                self.can_split = True
        if len(player_cards) > 2:
            self.can_double = False

        # Track Aces and add totals
        for card in player_cards:
            if card == "11":
                ace_count = ace_count + 1

            total = total + int(card)
        # Account for proper aces
        for ace in range(ace_count):
            if total > 21:
                ace_count = ace_count - 1
                total = total - 10
        # Determine if soft
        if ace_count == 1:
            is_soft = True

        return (total, is_soft)

    def set_true_count(self, true_count):
        self.true_count = true_count

    def get_action(self, upcard, player_cards, just_split=False):
        """Take player cards and dealer upcard return basic stratgy action"""
        self.upcard = upcard
        self.can_split = False
        card_list = []
        for card in player_cards:
            if card in ["J", "Q", "K"]:
                card = "10"
            if card == "A":
                card = "11"
            card_list.append(card)
        self.player_cards = card_list

        total, is_soft = self.get_total()
        self.total = total
        # if card1 == "11" or card2 == 11:
        #     is_soft = True

        if self.can_split and not just_split:
            self.handle_split()
            self.can_split = False
        if self.action != "SPLIT":
            if is_soft:
                self.handle_soft()
            else:
                self.handle_hard()

        action = self.action
        self.action = ""

        return action


if __name__ == "__main__":
    basicstrat = BasicStrategy()
    # This gon have to be a slice
    # list2 = []
    # totals = []
    # player_card_list = [
    #     ["2", "2"],
    #     ["3", "3"],
    #     ["4", "4"],
    #     ["5", "5"],
    #     ["6", "6"],
    #     ["7", "7"],
    #     ["8", "8"],
    #     ["9", "9"],
    #     ["10", "10"],
    #     ["11", "11"],
    #     ["2", "11"],
    #     ["3", "11"],
    #     ["4", "11"],
    #     ["5", "11"],
    #     ["6", "11"],
    #     ["7", "11"],
    #     ["8", "11"],
    #     ["9", "11"],
    #     ["10", "11"],
    #     ["3", "5"],
    #     ["4", "5"],
    #     ["4", "6"],
    #     ["6", "5"],
    #     ["7", "5"],
    #     ["8", "5"],
    #     ["9", "5"],
    #     ["10", "5"],
    #     ["10", "6"],
    #     ["10", "7"],
    # ]
    # dealer_upcard = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

    # # print(len(player_card_list))

    # for upcard in dealer_upcard:
    #     for pair in player_card_list:
    #         action = basicstrat.get_action(upcard, pair)
    #         if basicstrat.total <= 8 or basicstrat.total >= 17:
    #             continue
    #         print("\n\n\n\n\n\n")
    #         print("Player:", pair[0], pair[1])
    #         print("Player Total:", basicstrat.total)
    #         print("Dealer:", upcard)
    #         print(action)
    basicstrat.set_true_count(1)
    action = basicstrat.get_action("6", ["6", "2"])

    print(action)
