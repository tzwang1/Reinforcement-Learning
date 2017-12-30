import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.env.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factors=1.0):
    """
    Value iteration Algorithm. For each sweep perform one iteration of policy evaluation,
    and one iteration of policy iteration

    Args:
        env. OpneAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a]
            env.nS is the number of states in the environment.
            env.nA is the number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamme discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeroes(env.nS)
    policy = np.zeros([env.nS, env.nA])

    # Implement!
    while True:
        delta = 0
        for s in env.nS:
            all_values = np.zeros(env.nA)
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    all_values[a] += action_prob * prob * (reward + discount_factor * V[next_state])
            
            V[s] = np.max(all_values)
            best_a = np.argmax(all_values)
            # Always take the best action
            policy[s, best_a] = 1.0
            # Calculate the dleta across all states seen so far
            delta = np.max(delta, np.abs(best_action_value - V[s]))
        
        if delta < theta:
            break
    return policy, V

    # Test the value function
    expected_v = np.array([0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)