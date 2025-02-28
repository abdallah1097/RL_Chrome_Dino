import time

class QLearningAgent:
    def __init__(self, game_env):
        self._game_env = game_env


    def start_game(self):
        """
        Starts the game by jumping once
        """
        self._game_env.jump()
        time.sleep(.5)
