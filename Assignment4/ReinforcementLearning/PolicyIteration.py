import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt

# env_name = 'FrozenLake8x8-v0'
# new_map = generate_random_map(size=30, p=0.9)
# env = gym.make(env_name)
# env.seed(3006)
# conveged_at_determined = []


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = 1.0, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = 1.0):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            conveged_at = i+1
            # print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy, conveged_at


if __name__ == '__main__':

    env_name = 'FrozenLake8x8-v0'
    new_map = generate_random_map(size=30, p=0.9)
    conveged_at_determined = []
    policy_average_score_determined = []
    conveged_at_stochastic = []
    policy_average_score_stochastic = []
    gamma_list =[]

    gamma_range = np.arange(0.1,1.0,0.1)

    for i in gamma_range:
        print('iteration', i)
        env = gym.make(env_name, is_slippery=False)
        env.seed(3006)
        gamma = i
        optimal_policy, converged_at = policy_iteration(env, gamma)
        scores = evaluate_policy(env, optimal_policy, gamma)

        gamma_list.append(gamma)
        conveged_at_determined.append(converged_at)
        policy_average_score_determined.append(np.mean(scores))


    for i in gamma_range:
        print('iteration', i)
        env = gym.make(env_name)
        env.seed(3006)
        gamma = i
        optimal_policy, converged_at= policy_iteration(env, gamma)
        scores = evaluate_policy(env, optimal_policy, gamma)

        conveged_at_stochastic.append(converged_at)
        policy_average_score_stochastic.append(np.mean(scores))


    print('gamma list = ', gamma_list)
    print('conveged_at= ', conveged_at_determined)
    print('Average scores = ', policy_average_score_determined)
    print('conveged_at(stochastic)= ', conveged_at_stochastic)
    print('Average scores(stochastic)= ', policy_average_score_stochastic)

    plt.plot(gamma_list,conveged_at_determined, linestyle='--', marker='o', color='b')
    plt.xlabel("Gamma")
    plt.ylabel("iteration #")
    plt.title("Deterministic: converged happen at iteration # vs gamma")
    plt.show()
    #
    plt.plot(gamma_list, conveged_at_stochastic, linestyle='--', marker='o', color='b')
    plt.xlabel("Gamma")
    plt.ylabel("iteration #")
    plt.title("Stochastic: converged happen at iteration # vs gamma")
    plt.show()
    #

    plt.plot(gamma_list, policy_average_score_determined, marker = 'o')
    plt.plot(gamma_list, policy_average_score_stochastic, marker='o')
    plt.legend(('deterministic', 'stochastic'),loc= "upper left")
    plt.xlabel("Gamma")
    plt.ylabel("policy average score")
    plt.title("policy average score vs gamma")
    plt.show()