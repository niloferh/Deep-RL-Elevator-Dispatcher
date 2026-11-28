
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import LiftSimEnvironment
from dqn_agent.agent import DQNAgent


# Training configuration

# NUM_EPISODES = 500
NUM_EPISODES = 10
# MAX_STEPS_PER_EPISODE = 1000
MAX_STEPS_PER_EPISODE = 300

LEARNING_RATE = 1e-3
GAMMA = 0.99

# BATCH_SIZE = 64
BATCH_SIZE = 32
BUFFER_CAPACITY = 100000
#LEARNING_STARTS = 1000
LEARNING_STARTS = 100

TARGET_UPDATE_FREQUENCY = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000

SEED = 42

CHECKPOINT_DIR = "project/checkpoints"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "dqn_elevator.pth"
)

PLOTS_DIR = "project/plots"


def get_epsilon(step):
    """Calculate epsilon using linear decay."""

    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return EPSILON_START + fraction * (
        EPSILON_END - EPSILON_START
    )


def plot_training_results(episode_rewards, episode_losses):
    """Plot and save reward and loss against episode"""

    os.makedirs(PLOTS_DIR, exist_ok=True)

    episodes = range(1, len(episode_rewards) + 1)

    # Reward vs Episode
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episode")
    plt.grid(True)

    plt.savefig(f"{PLOTS_DIR}/dqn_reward_vs_episode.png")
    plt.show()
    plt.close()

    # Loss vs Episode
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, episode_losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title("Loss vs Episode")
    plt.grid(True)

    plt.savefig(f"{PLOTS_DIR}/dqn_loss_vs_episode.png")
    plt.show()
    plt.close()


def train():

    # Initialize environment

    env = LiftSimEnvironment(render=False)

    initial_observation = env.reset(seed=SEED)

    states = env.get_observation()

    state_dim = len(states[0])
    action_dim = env.action_dim

    print("Environment initialized.")
    print("Number of elevators:", env.num_elevators)
    print("Number of floors:", env.num_floors)
    print("State dimension:", state_dim)
    print("Action dimension:", action_dim)

    # Initialize DQN agent

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_frequency=TARGET_UPDATE_FREQUENCY
    )

    print("Using device:", agent.device)

    # Training

    total_steps = 0
    episode_rewards = []
    episode_losses = []

    for episode in range(NUM_EPISODES):

        # Reset environment at the beginning of each episode
        env.reset(seed=SEED + episode)

        states = env.get_observation()

        episode_reward = 0.0
        current_episode_losses = []

        for step in range(MAX_STEPS_PER_EPISODE):

            # Calculate exploration rate.
            epsilon = get_epsilon(total_steps)

            # Select one action per elevator
            actions = [
                agent.select_action(state, epsilon)
                for state in states
            ]

            # Execute all elevator actions together
            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(actions)

            next_states = env.get_observation()

            # Define the end of episode
            done = (
                terminated
                or truncated
                or step == MAX_STEPS_PER_EPISODE - 1
            )

            # Store one transition per elevator
            for i in range(env.num_elevators):

                agent.store_transition(
                    state=states[i],
                    action=actions[i],
                    reward=reward,
                    next_state=next_states[i],
                    done=done
                )

            # Learn once per simulation step after warm-up
            if len(agent.replay_buffer) >= LEARNING_STARTS:

                loss = agent.learn()

                if loss is not None:
                    current_episode_losses.append(loss)

            # Move to the next state
            states = next_states

            episode_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        # Calculate average loss for this episode
        average_loss = (
            np.mean(current_episode_losses)
            if current_episode_losses else 0.0
        )
        episode_losses.append(average_loss)

        # Print training progress

        if (episode + 1) % 10 == 0:

            average_reward = np.mean(
                episode_rewards[-10:]
            )

            recent_average_loss = np.mean(episode_losses[-10:])

            print(
                f"Episode {episode + 1}/{NUM_EPISODES} | "
                f"Steps: {total_steps} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Average Reward: {average_reward:.5f} | "
                f"Average Loss: {recent_average_loss:.5f}"
            )


    # Save trained model

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    torch.save(
        {
            "policy_network": agent.policy_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim
        },
        CHECKPOINT_PATH
    )

    plot_training_results(episode_rewards, episode_losses)

    print("\nTraining completed.")
    print("Model saved to:", CHECKPOINT_PATH)

    env.close()

    return episode_rewards


if __name__ == "__main__":
    train()
