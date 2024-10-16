import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper

# Environment setup
env = FullyObsWrapper(gym.make("MiniGrid-Fetch-8x8-N3-v0", render_mode="human"))
observation, info = env.reset(seed=42)

# Buffer to store transitions
demo_buffer = []

# Main loop to collect transition data from user input
terminated = False
for action in [0,2,2,2,1,3]:
    if action not in [0, 1, 2, 3, 4, 5, 6]:
        logger.info("Invalid action. Please enter 0, 1, 2, 3, 4, 5, 6.")
        continue
    #input("Press Enter to execute action {}".format(action))
    next_observation, reward, terminated, truncated, info = env.step(action)
    import ipdb; ipdb.set_trace()

    # Store the transition (obs1, action, obs2) in the buffer
    demo_buffer.append((observation['image'], action, next_observation['image']))

    observation = next_observation

    if terminated or truncated:
        observation, info = env.reset(seed=42)

assert terminated, "The episode should be terminated at the end of the demonstration."
