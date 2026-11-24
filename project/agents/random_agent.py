import random

from project.agents.utils.common_functions import (
    get_active_calls,
    remove_completed_assignments,
    generate_action
)

class RandomAgent:
    def __init__(self, num_elevators, seed=None):
        '''
        Initialize the random agent with the following attributes:
        - num_elevators: number of elevators in the environment
        - num_floors: number of floors in the environment
        - seed: random seed for reproducibility during testing (optional)
        '''
        
        self.num_elevators = num_elevators
        self.random_num = random.Random(seed)  # Use a local random generator for reproducibility
        self.assignments = {}  # Dictionary to keep track of elevator assignments to calls

    def act(self, observation):
        '''
        Given an obervation:
        - identify active calls (i.e., floors with waiting passengers
        - randomly select an elevator to service the call
        - create and return the action list
            action list = list of integers length 2*num_elevators where each pair of numbers represents an elevator 
            [dispatch_target, elevator_direction]
            - dispatch_target: the floor to which the elevator is dispatched (between 1 and maxfloornumber-10)
            - elevator_direction: the direction in which the elevator should move (-1=down, 0=no direction, 1=up)
        '''
        
        # get active calls from the observation
        activeCalls = get_active_calls(observation)
        
        
        # 
        self.assignments = remove_completed_assignments(self.assignments, activeCalls)
        
        # randomly assign new hall calls
        for call in activeCalls:
            if call not in self.assignments:
                self.assignments[call] = self.random_num.randrange(self.num_elevators)

        action = generate_action(self.assignments, self.num_elevators)
        return action
