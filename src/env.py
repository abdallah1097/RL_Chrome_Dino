
class DinoGame:
    def __init__():
        pass

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        pass