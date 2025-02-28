import time
import cv2
import pandas as pd
import os
from PIL import ImageGrab
import numpy as np
import pickle
from collections import deque
from src.model import QLearningDLModel
import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
import random


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
        self.num_actions = 2  # Two actions: [0: do nothing, 1: jump]
        self.learning_rate = 1e-4
        self.img_cols = 20
        self.img_rows = 40
        self.observe_timestamps = 50000  # Timestamps before training (Getting experience)
        self.frame_per_action = 1
        self.initial_epsilon = 0.1  # Initial Value of Epsilon
        self.final_epsilon = 0.0001  # Final Value of Epsilon
        self.explore_num_frames = 100000  # Frames over which to have epsilon (Exploration)



        # Initialize Experience Queue
        self.dqueue = deque()

        # Initialize Game Env attribute
        self._game_env = game_env
        self.jump()
        time.sleep(.5)  # Wait for the game to start

        # Initialize Display instance
        self._display = self.show_img()
        self._display.__next__() # initiliaze the display coroutine 

        # Intialize log structures from file if exists else create new
        self.loss_df = pd.read_csv(self.loss_file_path) if os.path.isfile(self.loss_file_path) else pd.DataFrame(columns =['loss'])
        self.scores_df = pd.read_csv(self.scores_file_path) if os.path.isfile(self.scores_file_path) else pd.DataFrame(columns = ['scores'])
        self.actions_df = pd.read_csv(self.actions_file_path) if os.path.isfile(self.actions_file_path) else pd.DataFrame(columns = ['actions'])

        # Write pickle files
        self.write_pickle(path=self.epsilon_file_path, value=self.initial_epsilon)
        self.write_pickle(path=self.time_file_path, value=0)
        self.write_pickle(path=self.dqueue_file_path, value=self.dqueue)

        # Define QLearning DL Model
        self.model = QLearningDLModel(
            img_channels=self.img_channels,
            num_actions=self.num_actions,
            img_cols=self.img_cols,
            img_rows=self.img_rows,
        ).to("cuda")
        summary(self.model, input_size=(1, self.img_channels, self.img_cols, self.img_rows))

        # Define Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

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
        return self._game_env.is_playing()

    def is_crashed(self):
        return self._game_env.is_crashed()

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

    def train(self):
        """
        Function to Train the agent
        """
        time_spent = 0
        epsilon = self.initial_epsilon

        # get the first state by doing nothing
        do_nothing = np.zeros(self.num_actions)
        do_nothing[0] = 1

        # Get initial state
        x_t, r_0, terminal = self.get_state(do_nothing)
        # Stack 4 images to create placeholder input
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # Reshape it to be 1*20*40*4

        # Assign Initial Reshaped State
        initial_state = s_t 

        while (True):
            action_index = 0
            a_t = np.zeros([self.num_actions])  # Initial Actions

            # Choose an epsilon-greedy action
            if time_spent % self.frame_per_action == 0: #parameter to skip frames for actions
                if  random.random() <= epsilon: #randomly explore an action
                    print("Picking Random Action")
                    action_index = random.randrange(self.num_actions)
                    a_t[0] = 1
                else:
                    torch_input = torch.from_numpy(s_t.transpose(0, 3, 1, 2)).float().to("cuda")
                    print(f"Picking Action from Model. States Shape: {torch_input.shape, torch_input.dtype}")
                    q = self.model(torch_input)  #input a stack of 4 images, get the prediction
                    q = q.detach().cpu().numpy()
                    max_Q = np.argmax(q) # chosing index with maximum q value
                    action_index = max_Q 
                    a_t[action_index] = 1  # o=> do nothing, 1=> jump

            # Reducing the epsilon (exploration parameter) gradually
            if epsilon > self.final_epsilon and time_spent > self.observe_timestamps:
                epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore_num_frames 
            print(f"Epsilon Value: {epsilon}")

