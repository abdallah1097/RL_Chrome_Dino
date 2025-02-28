from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import time

class DinoGame:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(options=chrome_options)
        self._driver.set_window_position(x=-10,y=0)
        self._driver.set_window_size(200, 300)
        self._driver.get(f"file://{os.path.abspath('src/dino_game/dino.html')}")  # Ensure it’s a file URL
        self._driver.execute_script("Runner.config.ACCELERATION=0")

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

