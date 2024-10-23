import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import heapq
import time
import logging
from tqdm import tqdm
from minigrid.wrappers import FullyObsWrapper
import pandas as pd

# Set up logging
logging.basicConfig(filename='agent_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

torch.manual_seed(42)

# Environment setup
env = FullyObsWrapper(gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")) #"MiniGrid-Fetch-8x8-N3-v0"
observation, info = env.reset(seed=42)

# Buffer to store transitions
demo_buffer = []

# Main loop to collect transition data from user input
terminated = False
for action in [2,2,1,2,2]: #[0,2,2,2,1,3]:
    if action not in [0, 1, 2, 3, 4, 5, 6]:
        logger.info("Invalid action. Please enter 0, 1, 2, 3, 4, 5, 6.")
        continue
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Store the transition (obs1, action, obs2) in the buffer
    demo_buffer.append((observation['image'], action, next_observation['image']))

    observation = next_observation

    if terminated or truncated:
        observation, info = env.reset(seed=42)

assert terminated

def test_agent_performance(agent, env, num_trials=5):
    """
    Function to test the agent and return the average number of steps to reach the goal using top-k plans.
    :param agent: AStarAgent instance.
    :param env: Environment instance to test the agent.
    :param num_trials: Number of trials to run for averaging.
    :return: Average number of steps to reach the goal.
    """
    total_steps = 0
    # Get top-k plans for the trial
    init_observation, info = env.reset(seed=42)
    start_state = agent.feature_extractor(torch.tensor(init_observation['image'], dtype=torch.float32).unsqueeze(0))
    top_k_plans = agent.get_top_k_plans(start_state, k=num_trials)
    if len(top_k_plans) == 0:
        return None
    for i in range(num_trials):
        observation, info = env.reset(seed=42)
        assert np.all(init_observation['image'] == observation['image'])
        terminated = False
        steps = 0

        # Use the best plan to execute in the environment
        best_plan, _ = top_k_plans[0]
        for state_tuple, action in best_plan[1:]:  # Skip the initial state
            if action is None:
                continue
            if terminated or steps >= 15:
                break
            observation, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated:
                logger.info("[SUCCESS] Goal reached during testing.")
                agent.generate_states_via_search("done.csv")
                break
        total_steps += steps
    return total_steps / num_trials

# Base Feature Extractor
class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space['image']
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

# Minigrid Feature Extractor
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space['image'].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space['image'].sample()[None]).float().permute(0, 3, 1, 2)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.Sigmoid())

        # Custom weight initialization for diversity
        nn.init.kaiming_uniform_(self.linear[0].weight, nonlinearity='relu')
        nn.init.uniform_(self.linear[0].bias, a=-0.1, b=0.1)

    def forward(self, observations):
        return self.linear(self.cnn(observations.float().permute(0, 3, 1, 2)))

# Model component
class TransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, next_state_dim):
        super(TransitionModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.next_state_dim = next_state_dim
        self.fc = nn.ModuleList([nn.Sequential(nn.Linear(state_dim, next_state_dim), nn.Sigmoid()) for _ in range(action_dim)])

    def forward(self, state, action):
        next_state = self.fc[action](state)
        return next_state

MAX_DIST = 1
def similarity_loss(state1, state2):
    return torch.max(1 - (nn.MSELoss()(state1, state2) / MAX_DIST), torch.tensor(0.0))

# A* Planning Agent
class AStarAgent:
    def __init__(self, feature_extractor, model, goal_state):
        self.feature_extractor = feature_extractor
        self.model = model
        self.goal_state = goal_state
        self.visited_nodes = []  # To store visited nodes for EPT Loss calculation
        self.max_planning_time = 0.05 # Maximum planning time in seconds
        self.atol = 5e-4 # Absolute tolerance for goal state

    def heuristic(self, state):
        return torch.norm(state - self.goal_state, p=2).item()

    def tensor_to_tuple(self, tensor):
        return tuple(tensor.detach().cpu().numpy().flatten())

    def a_star_search(self, start_state, verbose=False):
        start_state_tuple = self.tensor_to_tuple(start_state)
        frontier = []
        heapq.heappush(frontier, (0, start_state_tuple))
        came_from = {}
        cost_so_far = {}
        came_from[start_state_tuple] = None
        cost_so_far[start_state_tuple] = 0

        start_time = time.time()
        nodes_explored = 0
        self.visited_nodes = [start_state]  # Track visited nodes

        while frontier:
            if time.time() - start_time > self.max_planning_time:
                if verbose:
                    logger.info(f"A* search terminated after {self.max_planning_time} seconds. Nodes explored: {nodes_explored}")
                break

            _, current_tuple = heapq.heappop(frontier)
            nodes_explored += 1
            current = torch.tensor(current_tuple, dtype=torch.float32)

            if similarity_loss(current, self.goal_state) > (1 - self.atol):
                if verbose:
                    logger.info(f"Goal reached. Nodes explored: {nodes_explored}")
                break

            for action in range(self.model.action_dim):  # Assuming 7 possible actions
                next_state = self.model(current, action)
                next_state_tuple = self.tensor_to_tuple(next_state)
                new_cost = cost_so_far[current_tuple] + 1  # Assuming uniform cost

                if next_state_tuple not in cost_so_far or new_cost < cost_so_far[next_state_tuple]:
                    cost_so_far[next_state_tuple] = new_cost
                    priority = new_cost + self.heuristic(next_state)
                    heapq.heappush(frontier, (priority, next_state_tuple))
                    came_from[next_state_tuple] = (current_tuple, action)
                    self.visited_nodes.append(next_state)  # Track visited nodes

        return came_from

    def get_top_k_plans(self, start_state, k, verbose=False):
        """
        Function to get the top k plans using A* search.
        :param start_state: The starting state for A* search.
        :param k: The number of top plans to return.
        :param verbose: Whether to print verbose output.
        :return: List of top k plans (each plan is a list of (state, action)).
        """
        start_state_tuple = self.tensor_to_tuple(start_state)
        frontier = []
        heapq.heappush(frontier, (0, [(start_state_tuple, None)], 0))  # (f(n), path, g(n))
        top_k_paths = []

        start_time = time.time()
        nodes_explored = 0

        while frontier and len(top_k_paths) < k:
            if time.time() - start_time > self.max_planning_time:
                if verbose:
                    logger.info(f"Top-k A* search terminated after {self.max_planning_time} seconds. Nodes explored: {nodes_explored}")
                break

            f, path, g = heapq.heappop(frontier)
            current_tuple = path[-1]
            current = torch.tensor(current_tuple[0], dtype=torch.float32)
            nodes_explored += 1

            if similarity_loss(current, self.goal_state) > (1 - self.atol):
                top_k_paths.append((path, g))
                continue

            for action in range(self.model.action_dim):
                next_state = self.model(current, action)
                next_state_tuple = self.tensor_to_tuple(next_state)
                new_g = g + 1  # Assuming uniform cost
                new_f = new_g + self.heuristic(next_state)
                new_path = path + [(next_state_tuple, action)]
                heapq.heappush(frontier, (new_f, new_path, new_g))

        return top_k_paths

    def get_path(self, start_state_tuple, end_state_tuple, came_from):
        path = []
        current = end_state_tuple
        while current != start_state_tuple:
            if came_from[current] is None:
                break  # No valid path exists to the start state
            prev, action = came_from[current]
            prev, action = came_from[current]
            path.append((prev, action))
            current = prev
        path.reverse()
        return path

    def get_action(self, observation, verbose=False):
        state = self.feature_extractor(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
        state_tuple = self.tensor_to_tuple(state)
        plan = self.a_star_search(state, verbose=False)
        # Extract the first action from the plan
        if state_tuple in plan and plan[state_tuple] is not None:
            _, action = plan[state_tuple]
            return action
        return env.action_space.sample()  # Random action if no plan found

    def generate_states(self, file_path="graph.csv"):
        inputs = {}
        # Initialize the base array
        base_array = np.array([[[2, 5, 0]] * 5] * 5, dtype=np.uint8)
        
        # Add inner 3x3 block with the default values [1, 0, 0]
        for i in range(1, 4):
            for j in range(1, 4):
                if i != 3 or j != 3:
                    base_array[i][j] = [1, 0, 0]
                else:
                    base_array[i][j] = [8, 1, 0]
        
        # Loop over each position in the 3x3 block (positions [1,1] to [3,3])
        for i in range(1, 4):
            for j in range(1, 4):
                for k in range(0, 4):
                    # Create a copy of the base array
                    modified_array = np.copy(base_array)
                    
                    # Swap the value at position (i, j) in the inner block
                    modified_array[i][j] = [10, k, 0]
                    
                    # Print the resulting array
                    inputs[f"{i}_{j}_{k}"] = modified_array

        # Initialize a list to store all states and a list for the corresponding keys
        states = []
        keys = []

        # Extract states from the inputs and store them with corresponding keys
        for key, val in inputs.items():
            state = self.feature_extractor(torch.tensor(val, dtype=torch.float32).unsqueeze(0))
            states.append(state)
            keys.append(key)
            for action in range(7):
                state = self.model(state, action)
                states.append(state)
                keys.append(key + f"_{action}")

        # Stack all states into a single tensor for pairwise distance computation
        states_tensor = torch.cat(states, dim=0)  # Shape: (N, feature_dim), where N is the number of states

        # Compute pairwise distances between all states
        pairwise_distances = torch.cdist(states_tensor, states_tensor)

        # Convert pairwise distances to a Pandas DataFrame with the keys as index and column names
        distance_df = pd.DataFrame(pairwise_distances.detach().numpy(), index=keys, columns=keys)

        # Display the DataFrame
        distance_df.to_csv(file_path)

    def generate_states_via_search(self, file_path="graph.csv"):
        states = []
        keys = []

        start_state = self.feature_extractor(torch.tensor(demo_buffer[0][0], dtype=torch.float32).unsqueeze(0))
        states.append(start_state)
        keys.append("start")
        states.append(self.goal_state)
        keys.append("goal")
        plans = self.a_star_search(start_state, verbose=True)
        # Extract states from the inputs and store them with corresponding keys
        i = 0
        for key, val in plans.items():
            state = torch.tensor(key, dtype=torch.float32).unsqueeze(0)
            states.append(state)
            keys.append(str(i))
            i += 1
            if val is None:
                continue
            from_state, action = val
            from_state = torch.tensor(from_state, dtype=torch.float32).unsqueeze(0)
            states.append(from_state)
            keys.append(str(i))
            states.append(state)
            keys.append(f"{i}_{action}")
            i += 1

        # Stack all states into a single tensor for pairwise distance computation
        states_tensor = torch.cat(states, dim=0)  # Shape: (N, feature_dim), where N is the number of states

        # Convert pairwise distances to a Pandas DataFrame with the keys as index and column names
        distance_df = pd.DataFrame(states_tensor.detach().numpy(), index=keys)

        # Display the DataFrame
        distance_df.to_csv(file_path)
        


def compute_uniformity_loss(embeddings, t=2):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2) ** 2
    mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()

    dist_matrix = dist_matrix.masked_fill(mask, float('inf'))
    loss = torch.exp(-t * dist_matrix)

    # Mean uniformity loss
    return loss.sum() / (embeddings.size(0) * (embeddings.size(0) - 1))

# Assuming `demo_buffer` is a list of (obs1, action, _) tuples
def get_uniformity_loss_from_buffer(demo_buffer, feature_extractor, model):
    embeddings = []

    for (obs1, action, _) in demo_buffer:
        # Extract features from obs1 using the feature extractor
        state1 = feature_extractor(torch.tensor(obs1, dtype=torch.float32).unsqueeze(0))
        predicted_next_state = model(state1, action)  # Shape: [batch_size=1, state_dim]
        embeddings.append(state1)
        embeddings.append(predicted_next_state)

    embeddings = torch.cat(embeddings, dim=0)  # Shape: [num_samples, state_dim]
    loss = compute_uniformity_loss(embeddings)

    return loss

# Load weights flag
LOAD_WEIGHTS = False

# Initialize components
state_dim = 64
obs_shape = observation['image'].shape
feature_extractor = MinigridFeaturesExtractor(observation_space=env.observation_space, features_dim=state_dim)
model = TransitionModel(state_dim=state_dim, action_dim=7, next_state_dim=state_dim)
goal_state = feature_extractor(torch.tensor(demo_buffer[-1][2], dtype=torch.float32).unsqueeze(0))  # Goal state is the state of the last observation in the demo
agent = AStarAgent(feature_extractor, model, goal_state)

# Load model weights if LOAD_WEIGHTS is True
if LOAD_WEIGHTS:
    try:
        model = torch.load('weights/model_weights.pt')
        feature_extractor = torch.load('weights/feature_extractor_weights.pt')
        logger.info("Weights loaded successfully.")
    except FileNotFoundError:
        logger.info("Weight files not found. Starting from scratch.")
    
# ETPT Loss Optimization
optimizer = optim.Adam(model.parameters(), lr=1e-3)
MAX_STEPS = 10
MAX_DEPTH = 10
BACKPROP_EVERY_N_EPOCHS = 1
UNIFORMITY_WEIGHT = 0.0 #1e-5
EPSILON = 1e-6
N = 5  # Number of sub-trajectories to sample
ETPT_loss = float('inf')

# Call the function to print all arrays
agent.generate_states_via_search()

for epoch in tqdm(range(20000)):  # Number of optimization epochs
    # Test the top k plans every epoch
    top_k_plans = agent.get_top_k_plans(feature_extractor(torch.tensor(demo_buffer[0][0], dtype=torch.float32).unsqueeze(0)), k=3)
    for idx, (plan, cost) in enumerate(top_k_plans):
        logger.info(f"Plan {idx+1}, Cost: {cost}, Length: {len(plan)}")
        logger.info(f"Plan: {[a for s, a in plan]}")
    if epoch % 1000 == 0:
        # Test the agent every epoch to see the average number of steps in the environment
        avg_steps = test_agent_performance(agent, env)
        logger.info(f"Epoch {epoch+1}, Average steps to goal: {avg_steps}")
        agent.generate_states_via_search(f"graph_{epoch}.csv")
        #input("Press Enter to continue...")

        # Save model weights
        torch.save(model, 'model_weights.pt')
        torch.save(feature_extractor, 'feature_extractor_weights.pt')
    if ETPT_loss < len(demo_buffer):
        logger.info("ETPT loss below threshold. Stopping training.")
        break
    total_loss = 0.0
    # Compute (ERP Loss) Expected Refinement Probability Loss
    log_ERP_loss = 0.0
    for end_i in range(1, len(demo_buffer)):
        obs1 = demo_buffer[0][0]
        state1 = feature_extractor(torch.tensor(obs1, dtype=torch.float32).unsqueeze(0))
        predicted_final_state = state1
        for j in range(0, end_i+1):
            action = demo_buffer[j][1]
            predicted_final_state = model(predicted_final_state, action)
        actual_final_state = feature_extractor(torch.tensor(demo_buffer[j][2], dtype=torch.float32).unsqueeze(0))
        log_ERP_loss += torch.log(similarity_loss(predicted_final_state, actual_final_state) + EPSILON)
    ERP_loss = torch.exp(log_ERP_loss)
    ERP_loss_step_loss = (1 - ERP_loss) * MAX_STEPS

    # Uniformity Loss to encourage diversity in predicted states
    uniformity_loss = get_uniformity_loss_from_buffer(demo_buffer, feature_extractor, model)
    ERP_loss_step_loss += torch.mul(uniformity_loss, UNIFORMITY_WEIGHT)

    if (epoch + 1) % BACKPROP_EVERY_N_EPOCHS != 0:
        # Backpropagation for ERP Loss every step
        optimizer.zero_grad()
        ERP_loss_step_loss.backward()
        optimizer.step()
    else:
        # Compute (EPT Loss) Expected Planning Time Loss
        start_state = feature_extractor(torch.tensor(demo_buffer[0][0], dtype=torch.float32).unsqueeze(0))
        EPT_loss = (1.0 - similarity_loss(start_state, agent.goal_state))
        plans = agent.a_star_search(start_state, verbose=True)
        for visited_state in agent.visited_nodes[:MAX_DEPTH]:
            path_to_node = agent.get_path(agent.tensor_to_tuple(torch.tensor(demo_buffer[0][0])), agent.tensor_to_tuple(visited_state), plans)
            path_probability = 1.0
            for (prev_state_tuple, action) in path_to_node:
                prev_state = torch.tensor(prev_state_tuple, dtype=torch.float32).unsqueeze(0)
                predicted_next_state = model(prev_state, action)
                path_probability *= (1.0 - similarity_loss(predicted_next_state, agent.goal_state))
            EPT_loss += path_probability

        # Compute (ETPT Loss) as the sum of ERP Loss and EPT Loss every BACKPROP_EVERY_N_EPOCHS epochs
        EPT_loss_match_demo = torch.exp(torch.abs(len(demo_buffer) + 1 - (EPT_loss)))
        ETPT_loss = EPT_loss_match_demo + ERP_loss_step_loss
        total_loss += ETPT_loss.item()

        # Backpropagation for ETPT Loss
        optimizer.zero_grad()
        ETPT_loss.backward()
        optimizer.step()


        # Update goal state after backpropagation
        agent.goal_state = feature_extractor(torch.tensor(demo_buffer[-1][2], dtype=torch.float32).unsqueeze(0))

        logger.info(f"Epoch {epoch+1}, EPT Loss: {EPT_loss_match_demo.item()}, ERP Loss: {ERP_loss_step_loss.item()}, ETPT Loss: {ETPT_loss.item()}")
        logger.info(f"ERP: {ERP_loss.item()}, EPT: {EPT_loss.item()}, Match Demo Length: {len(demo_buffer)}, Uniformity: {uniformity_loss.item()}")

# Test the agent on the environment
test_observation, info = env.reset(seed=42)
terminated = False
while not terminated:
    action = agent.get_action(test_observation['image'])
    test_observation, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        logger.info("Goal reached during testing.")
        break

env.close()