import time
import cv2
import pandas as pd
import os
from PIL import ImageGrab
import numpy as np
import pickle
from collections import deque
from src.model import QLearningDLModel


class QLearningAgent:
    def __init__(self, game_env):
        # Initialize df paths
        self.loss_file_path = "./objects/loss_df.csv"
        self.actions_file_path = "./objects/actions_df.csv"
        self.scores_file_path = "./objects/scores_df.csv"
        self.epsilon_file_path = "./objects/epsilon.pkl"
        self.time_file_path = "./objects/time.pkl"
        self.dqueue_file_path = "./objects/queue.pkl"

        # Model Parameters
        self.img_channels = 4
        self.num_actions = 2
        self.learning_rate = 1e-4
        self.img_cols = 20
        self.img_rows = 40

        # Initialize Experience Queue
        self.dqueue = deque()

        # Initialize Game Env attribute
        self._game_env = game_env

        # Initialize Display instance
        self._display = self.show_img()
        self._display.__next__() # initiliaze the display coroutine 

        # Intialize log structures from file if exists else create new
        self.loss_df = pd.read_csv(self.loss_file_path) if os.path.isfile(self.loss_file_path) else pd.DataFrame(columns =['loss'])
        self.scores_df = pd.read_csv(self.scores_file_path) if os.path.isfile(self.scores_file_path) else pd.DataFrame(columns = ['scores'])
        self.actions_df = pd.read_csv(self.actions_file_path) if os.path.isfile(self.actions_file_path) else pd.DataFrame(columns = ['actions'])

        # Write pickle files
        self.write_pickle(path=self.epsilon_file_path, value=0.1)
        self.write_pickle(path=self.time_file_path, value=0)
        self.write_pickle(path=self.dqueue_file_path, value=self.dqueue)

        # Define QLearning DL Model
        self.model = QLearningDLModel(
            self.img_channels,
            self.num_actions,
            self.learning_rate, self.img_cols,
            self.img_rows,
        )
        print(f"Model: {self.model}")

    def write_pickle(self, path, value):
        with open(path, 'wb') as f: #dump files into objects folder
            pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)

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

    def get_state(self, actions):
        """
        Captures the current game state, processes the screen image, and calculates rewards.

        Args:
            actions (list): A list representing the actions taken. The second element (actions[1]) 
                            determines whether the Dino jumps.

        Returns:
            tuple: (image, reward, is_over)
                - image (numpy.ndarray): Processed game screen.
                - reward (float): Calculated reward based on the score.
                - is_over (bool): True if the game is over (Dino has crashed), False otherwise.
        """
        self.actions_df.loc[len(self.actions_df)] = actions[1] # storing actions in a dataframe
        score = self._game_env.get_score() 
        reward = 0.1 * score / 10  # Dynamic reward calculation
        is_over = False  # Game over flag

        if actions[1] == 1:
            self.jump()
            reward = 0.1 * score / 11

        image = self.grab_screen()  # Capture current screen
        self._display.send(image)  # Display the image on screen

        if self.is_crashed():
            self.scores_df.loc[len(self.loss_df)] = score  # Log the score when game is over
            self._game_env.restart()
            reward = -11/score
            is_over = True

        return image, reward, is_over #return the Experience tuple

    def grab_screen(self):
        """
        Captures a screenshot of the game region and processes it.

        Returns:
            numpy.ndarray: Processed game screen image.
        """
        screen = np.array(ImageGrab.grab(bbox=(40,180,440,400)))  # Define region of interest
        image = self.process_img(screen)  # Process the captured image
        return image

    def process_img(self, image):
        """
        Processes the game screen image using resizing, cropping, and edge detection.

        Args:
            image (numpy.ndarray): The raw screen capture.

        Returns:
            numpy.ndarray: The processed grayscale image with edge detection applied.
        """
        image = cv2.resize(image, (0, 0), fx=0.15, fy=0.10)  # Resize image
        image = image[2:38, 10:50]  # Crop out the Dino agent from the frame
        image = cv2.Canny(image, threshold1=100, threshold2=200)  # Apply Canny edge detection
        return image
