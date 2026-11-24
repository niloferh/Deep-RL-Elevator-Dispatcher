from metagym.liftsim.environment.env import LiftSim
from metagym.liftsim.environment.mansion import mansion_manager
from project.agents.random_agent import RandomAgent

# initialize environment and agent
env = LiftSim(config_file = r"C:\Users\khale\Desktop\ECE1508 Project\MetaGym\project\configs\config_three_elevators.ini")
env = LiftSim(config_file = r"C:\Users\khale\Desktop\ECE1508 Project\MetaGym\project\configs\config_three_elevators.ini")
env = LiftSim(config_file = r"C:\Users\khale\Desktop\ECE1508 Project\MetaGym\project\configs\config_three_elevators.ini")


observation = env.reset()

agent = RandomAgent(
    num_elevators=6,
    seed=42
)

# run simulation
totalReward = 0
totalTimeConsumed = 0
totalEnergyConsumed = 0
totalAbandonedPassengers = 0
done = False
step = 0

#for step in range(500):
while not done:
    action = agent.act(observation)

    #action = [0] * (2 * 6)  # Initialize action list for 6 elevators
    observation, reward, done, info = env.step(action)

    totalReward += reward
    totalTimeConsumed += info['time_consume']
    totalEnergyConsumed += info['energy_consume']
    totalAbandonedPassengers += info['given_up_persons']
    step += 1

    env.render()

    print(f"Step: {step}")
    print(f"Action: {action}") 
    print(f"Reward: {reward}")
    print(info)

    if done or step >= 999:
        break

# print(f"\nTotal reward: {totalReward}")
# print(f"Total time consumed: {totalTimeConsumed}")
# print(f"Total energy consumed: {totalEnergyConsumed}")
# print(f"Total abandoned passengers: {totalAbandonedPassengers}")

stat = env._mansion.get_statistics()


print(stat)

env.close()