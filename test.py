from snake_env import SnakeEnv
from stable_baselines3 import PPO 

env = SnakeEnv()
model = PPO.load("ppo_snake", env=env)

obs = env.reset()
done = False
env.render()
while not done:
    # input("press enter to continue")
    action, _info = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
env.close()
