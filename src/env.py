from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


import os
import time

class DinoGameEnv:
    def __init__(self):
        """
        Initializes a Chrome WebDriver instance for running the Dino game.

        This constructor sets up the Chrome WebDriver with specific options, 
        positions and resizes the browser window, and loads the local Dino game HTML file. 
        Additionally, it disables acceleration in the game using JavaScript.

        Attributes:
            _driver (webdriver.Chrome): The Chrome WebDriver instance.
        """
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(300, 500)
        self._driver.get(f"file://{os.path.abspath('src/dino_game/dino.html')}")  # Ensure it’s a file URL
        self._driver.execute_script("Runner.config.ACCELERATION=0")

    def is_crashed(self):
        """
        Checks if the Dino game character has crashed.

        Returns:
            bool: True if the game is over (Dino has crashed), False otherwise.
        """
        return self._driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        """
        Checks if the Dino game is currently running.

        Returns:
            bool: True if the game is in progress, False if it is paused or over.
        """
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        """
        Restarts the Dino game by triggering the restart function in the game script.
        """
        self._driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.25)  # Sleep to let game restart

    def jump(self):
        """
        Makes the Dino character jump by simulating an 'Arrow Up' key press.
        """
        element = self._driver.find_element(By.TAG_NAME, "body")
        element.send_keys(Keys.ARROW_UP)

    def duck(self):
        """
        Makes the Dino character duck by simulating an 'Arrow Down' key press.
        """
        element = self._driver.find_element(By.TAG_NAME, "body")
        element.send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        """
        Retrieves the current score of the game.

        Returns:
            int: The player's current score as an integer.
        """
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        return int(score)

    def pause(self):
        """
        Pauses the Dino game.

        Returns:
            None
        """
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        """
        Resumes the Dino game after being paused.

        Returns:
            None
        """
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        """
        Closes the WebDriver instance, effectively ending the game session.
        """
        self._driver.close()
