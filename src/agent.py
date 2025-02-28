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

    def is_running(self):
        return self._game_env.get_playing()

    def is_crashed(self):
        return self._game_env.get_crashed()

    def jump(self):
        self._game_env.jump()

    def duck(self):
        self._game_env.duck()
