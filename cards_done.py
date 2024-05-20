

def can_add(cards):
    """Take values and add them account for aces"""
    total = 0
    for card in cards:
        if card in ["J", "Q", "K"]:
            card = "10"
        if card == "A":
            card = "1"

        total = total + int(card)

    if total <= 20:
        return True
    else:
        return False


def get_card_limit(cards):
    """If all possible player actions have occered return True to stop card_slot"""
    if len(cards) <= 2:
        return False
    cards_formatted = []
    cards_bools = []
    cards_handled = True
    hand1_done = False
    hand2_done = True
    hand1 = []
    hand2 = []
    for card in cards:
        if card == "A":
            card = "11"
        if card in ["J", "Q", "K"]:
            card = "10"
        cards_formatted.append(card)
        cards_bools.append(False)

    # Will split
    split = False

    if cards_formatted[0] == cards_formatted[1]:
        split = True

    stop_cards = False
    if split:
        hand1.append(cards_formatted[0])
        hand2.append(cards_formatted[1])
        cards_formatted.pop(0)
        cards_formatted.pop(0)
        for card in cards_formatted:
            if not hand1_done:
                adding = can_add([*hand1,card])
                if not adding:
                    # IF 21 or more
                    hand1_done = True
                    hand2_done = False
                    hand1.append(card)
                else:
                    hand1.append(card)
            elif not hand2_done:
                adding = can_add([*hand2,card])
                if not adding:
                    # IF 21 or more
                    hand2_done = True
                    hand2.append(card)
                    stop_cards = True
                else:
                    hand2.append(card)
    else:
        for card in cards_formatted:
            adding = can_add(hand1)
            if not adding:
                # IF 21 or more
                hand1_done = True
                hand1.append(card)
                stop_cards = True

            else:
                hand1.append(card)
    
    print(hand1)
    print(hand2)
    print(stop_cards)
        # HANDLE SINGLE HAND

cards = ["Q", "K", "10", "2", "10", "10"]

get_card_limit(cards)
