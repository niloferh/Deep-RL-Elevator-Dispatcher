def get_active_calls(observation):
    '''
    Given an observation, return a list of active calls (i.e., floors with waiting passengers) and directions
    '''
    activeCalls = set()

    for floor in observation.RequiringUpwardFloors:
        activeCalls.add((floor, 1))

    for floor in observation.RequiringDownwardFloors:
        activeCalls.add((floor, -1))

    return activeCalls


def remove_completed_assignments(assignments, activeCalls):
    """Remove assignments for calls that are no longer active."""

    return {
        call: elevator
        for call, elevator in assignments.items()
        if call in activeCalls
    }


def generate_action(assignments, numElevators):
    """Convert elevator assignments into a LiftSim action."""

    action = [0] * (2 * numElevators)

    for elevator in range(numElevators):

        assignedCalls = [
            call
            for call, assignedElevator in assignments.items()
            if assignedElevator == elevator
        ]

        if assignedCalls:
            floor, direction = assignedCalls[0]

            action[2 * elevator] = floor
            action[2 * elevator + 1] = direction

    return action