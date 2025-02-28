from src.agent import QLearningAgent
from src.env import DinoGameEnv


if __name__ == "__main__":
    env = DinoGameEnv()
    agent = QLearningAgent(env)

    agent.train()