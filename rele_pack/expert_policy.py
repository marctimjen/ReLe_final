class expert_policy:
    def __init__(self):
        self.collected_coin = False

    def play_step(self, state):
        if not(self.collected_coin):
            coin_diff = state[6:8]
            x, y = None, None
            if coin_diff[1] > 0.0001:  # 0 = up, 2 = down
                x = 2
            elif coin_diff[1] < -0.0001:
                x = 0
            else:
                x = 1

            if coin_diff[0] > 0.0001:  # 0 = left, 2 = right
                y = 2
            elif coin_diff[0] < -0.0001:
                y = 0
            else:
                y = 1

            if x == 1 and y == 1:
                self.collected_coin = True
                x, y = self.play_step(state)

            return x, y

        else:
            chest_diff = state[8:10]
            x, y = None, None

            if chest_diff[1] > 0.0001:
                x = 2
            elif chest_diff[1] < -0.0001:
                x = 0
            else:
                x = 1

            if chest_diff[0] > 0.0001:
                y = 2
            elif chest_diff[0] < -0.0001:
                y = 0
            else:
                y = 1

            return x, y
