'''FrozenLake'''

import numpy as np
import pandas as pd
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt


def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
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


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-10
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.nS):
            q_sa = [sum([p*(r + gamma* prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            convegred_at = i+1
            # print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v, convegred_at

def draw_env(env):
    shape = env.desc.shape
    row = shape[0]
    column = shape[1]
    arr = np.zeros(shape)

    for i in range(row):
        for j in range(column):
            if env.desc[i,j] == b'H':
                arr[i, j] = 0.25
            elif env.desc[i,j] == b'G':
                arr[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap='summer')
    ax.set_xticks(np.arange(row))
    ax.set_yticks(np.arange(column))
    ax.set_xticklabels(np.arange(row))
    ax.set_yticklabels(np.arange(column))
    ax.set_xticks(np.arange(-0.5, row, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, column, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(row):
        for j in range(column):
            if (i, j) == (0, 0):
                ax.text(j, i, 'Start', ha='center', va='center', color='k', size=12)
            if env.desc[i,j] == b'H':
                ax.text(j, i, 'H', ha='center', va='center', color='k', size=12)
            elif env.desc[i,j] == b'G':
                ax.text(j, i, 'G', ha='center', va='center', color='k', size=14)
            else:
                pass
    fig.tight_layout()
    plt.show()

def draw_policy(policy, env):
    shape = env.desc.shape
    row = shape[0]
    column = shape[1]
    policy = np.array(policy)
    actions = policy.reshape(shape)
    mapping = {
        0: '<-',
        1: 'v',
        2: '->',
        3: '^'
    }
    arr = np.zeros(shape)
    for i in range(row):
        for j in range(column):
            if env.desc[i,j] == b'H':
                arr[i, j] = 0.25
            elif env.desc[i,j] == b'G':
                arr[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap='summer')
    ax.set_xticks(np.arange(row))
    ax.set_yticks(np.arange(column))
    ax.set_xticklabels(np.arange(row))
    ax.set_yticklabels(np.arange(column))
    ax.set_xticks(np.arange(-0.5, row, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, column, 1), minor=True)
    ax.grid(False)
    ax.grid(which='minor', color='w', linewidth=2)

    for i in range(row):
        for j in range(column):
            if env.desc[i,j] == b'H':
                ax.text(j, i, 'H', ha='center', va='center', color='k', size=12)
            elif env.desc[i,j] == b'G':
                ax.text(j, i, 'G', ha='center', va='center', color='k', size=14)
            else:
                ax.text(j, i, mapping[actions[i, j]], ha='center', va='center', color='k', size=12)
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    env_name  = 'FrozenLake8x8-v0'
    new_map = generate_random_map(size = 30, p =0.9)
    conveged_at_determined = []
    policy_average_score_determined = []
    conveged_at_stochastic = []
    policy_average_score_stochastic = []
    gamma_list =[]


    gamma_range = np.arange(0.1,1.0,0.1)

    for i in gamma_range:

        gamma = 1-i/100
        gamma_list.append(gamma)

        """ deterministic """
        env1 = gym.make(env_name, is_slippery=False)
        env1.seed(3006)

        optimal_v, convegred_at= value_iteration(env1, gamma)
        conveged_at_determined.append(convegred_at)
        policy = extract_policy(env1, optimal_v, gamma)
        # print('Policy = ', policy)
        policy_score = evaluate_policy(env1, policy, gamma, n=1000)
        policy_average_score_determined.append(policy_score)

        """ stochastic """
        env2 = gym.make(env_name, desc = new_map)
        env2.seed(3006)
        optimal_v2, convegred_at2= value_iteration(env2, gamma)
        conveged_at_stochastic.append(convegred_at2)
        policy2 = extract_policy(env2, optimal_v2, gamma)
        # print('Policy = ', policy)
        policy_score2 = evaluate_policy(env2, policy2, gamma, n=1000)
        policy_average_score_stochastic.append(policy_score2)

    print('Gamma =', gamma_list)
    print('Policy converged at = ', conveged_at_determined)
    print('Policy average score = ', policy_average_score_determined)
    plt.plot(gamma_list,conveged_at_determined, linestyle='--', marker='o', color='b')
    plt.xlabel("Gamma")
    plt.ylabel("iteration #")
    plt.title("Deterministic: converged happen at iteration # vs gamma")
    plt.show()

    plt.plot(gamma_list,conveged_at_stochastic, linestyle='--', marker='o', color='b')
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

    # env1 = gym.make(env_name)
    # env1.seed(3006)
    # # env1.render()
    # shape = env1.desc.shape
    # # draw_env(env1)
    # gamma = 0.99
    # optimal_v, convegred_at= value_iteration(env1, gamma)
    # conveged_at_determined.append(convegred_at)
    # policy = extract_policy(env1, optimal_v, gamma)
    # # print('Policy = ', policy)
    # policy_score = evaluate_policy(env1, policy, gamma, n=1000)
    # policy_average_score_determined.append(policy_score)
    # draw_policy(policy, env1)
