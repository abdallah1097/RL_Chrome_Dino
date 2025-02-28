import time
import cv2


class QLearningAgent:
    def __init__(self, game_env):
        # Initialize Game Env attribute
        self._game_env = game_env

        # Initialize Display instance
        self._display = self.show_img()
        self._display.__next__() # initiliaze the display coroutine 

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

    def show_img(self, graphs=False):
        """
        Shows the processed images in new window (on-screen) using openCV, implemented using python coroutine
        """
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
            imS = cv2.resize(screen, (800, 400)) 
            cv2.imshow(window_title, screen)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                cv2.destroyAllWindows()
                break
