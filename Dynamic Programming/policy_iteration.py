import numpy as np
import pprint
import sys
if "../" not in sys.path:
    sys.path.append("../")
from lib.env.gridworld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

# Taken from Policy Evaluation Exercise!
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.

        discount_factor: Gamma discount factor
        .
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(V - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)

    def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
        """
        Policy Improvement Algorithm. Iteratively evaluates and improves a policy
        until an optimal policy is found.

        Args:
            env: The OpenAI environment.
            policy_eval_fun: Policy Evaluation function that takes 3 argument:
                policy, env, discount_factor
            discount_factor: gamma discount factor
        
        Returns:
            A tuple (policy, V).
            policy is the optimal policy, a matrix of shape [S, A] where each state s
            contains a valid probability distribution over actions.
            V is the value function for the optimal policy.
        """
        # Start with a random policy
        policy = np.ones([env.nS, env.nA]) / env.nA
        
        while True:
            # Implement this!
            V = policy_eval_fn(policy, env)
            policy_stable = True
            for s in range(env.nS):
                # The best action we would take under the current policy
                chosen_a = np.argmax(policy[s])
                
                # Find the best action by one-step lookahead
                # Ties are resolved arbitrarily
                action_values = np.zeros(env.nA)
                #for a in range(env.nA):
                for a, action_prob in enumerate(policy[s]):
                    for prob, next_state, reward, done in env.P[s][a]:
                        action_values[a] += action_prob * prob * (reward + discount_factor * V[next_state])
                best_a = np.argmax(action_values)
                    
                # Greedily update the policy
                if chosen_a != best_a:
                    policy_stable = False
                policy[s] = np.eye(env.nA)[best_a]

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V

    policy, v = policy_improvement(env)
    
    # Test the value function
    expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
