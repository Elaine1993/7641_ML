import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import ValueIteration
from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.mdp import QLearning


import matplotlib.pyplot as plt

fontsize = 15

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
    #print("---------")
    #print(step_idx)
    #print(obs)
    print(reward)
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

def iter_score(vi, vi2):
    print("iter_score")
    v1_mean = [v["Mean V"] for v in vi.run_stats]
    v2_mean = [v["Mean V"] for v in vi2.run_stats]
    v2_mean.extend([v2_mean[-1]] * (len(v1_mean)-len(v2_mean)))

    r = range(len(v1_mean))

    plt.plot(r, v1_mean, linestyle='--', marker='o', color='b')
    plt.plot(r, v2_mean, linestyle='--', marker='o', color='r')
    plt.xlabel("Iteration #")
    plt.ylabel("Mean score")
    plt.title("Mean score(reward) vs iteration")
    plt.legend(('fire possibility = 0.1', 'fire possibility = 0.9'), loc="upper left")
    plt.show()



def gamma_iter_value():
    gamma = np.arange(0.1, 1.0, 0.1)
    v1_iter = []
    v2_iter = []
    v1_v_mean = []
    v2_v_mean = []
    for g in gamma:
        P, R = forest(num_states, r1, r2, p_fire)
        P2, R2 = forest(num_states, r1, r2, 0.9)
        vi = ValueIteration(P, R, g, 1e-20)
        vi.run()

        vi2 = ValueIteration(P2, R2, g, 1e-20)
        vi2.run()
        v1_iter.append(len(vi.run_stats))
        v2_iter.append(len(vi2.run_stats))
        v1_v_mean.append(vi.run_stats[-1]["Mean V"])
        v2_v_mean.append(vi2.run_stats[-1]["Mean V"])

    # plt.plot(gamma, v1_iter, linestyle='--', marker='o', color='b',label="fire possibility = 0.1")
    # plt.plot(gamma, v2_iter, linestyle='--', marker='o', color='r',label="fire possibility = 0.9")
    # plt.xlabel("Gamma")
    # plt.ylabel("Converged iteration #")
    # plt.title("converged happen at iteration # vs gamma")
    # plt.legend(('fire possibility = 0.1', 'fire possibility = 0.9'), loc="upper left")
    # plt.show()


    plt.plot(gamma,v1_v_mean, linestyle='--', marker='o', color='b',label="fire possibility = 0.1")
    plt.plot(gamma, v2_v_mean, linestyle='--', marker='o', color='r',label="fire possibility = 0.9")
    plt.xlabel("Gamma")
    plt.ylabel("Converged Mean Value")
    plt.title("converged Mean Value vs gamma")
    plt.legend(('fire possibility = 0.1', 'fire possibility = 0.9'), loc="upper left")
    plt.show()




def gamma_iter_value_p():
    gamma = np.arange(0.1, 1.0, 0.1)
    v1_iter = []
    v2_iter = []
    v1_v_mean = []
    v2_v_mean = []
    for g in gamma:
        P, R = forest(num_states, r1, r2, p_fire)
        P2, R2 = forest(num_states, r1, r2, 0.9)
        vi = PolicyIteration(P, R, g)
        vi.run()

        vi2 = PolicyIteration(P2, R2, g)
        vi2.run()
        v1_iter.append(len(vi.run_stats))
        v2_iter.append(len(vi2.run_stats))
        v1_v_mean.append(vi.run_stats[-1]["Mean V"])
        v2_v_mean.append(vi2.run_stats[-1]["Mean V"])

    # plt.plot(gamma,v1_iter, linestyle='--', marker='o', color='b',label="fire possibility = 0.1")
    # plt.plot(gamma, v2_iter, linestyle='--', marker='o', color='r',label="fire possibility = 0.9")
    # plt.xlabel("Gamma")
    # plt.ylabel("Converged iteration #")
    # plt.title("Converged happen at iteration# vs gamma")
    # plt.legend(('fire possibility = 0.1', 'fire possibility = 0.9'), loc="upper left")
    # plt.show()



    plt.plot(gamma,v1_v_mean, linestyle='--', marker='o', color='b',label="fire possibility = 0.1")
    plt.plot(gamma, v2_v_mean, linestyle='--', marker='o', color='r',label="fire possibility = 0.9")
    plt.xlabel("Gamma")
    plt.ylabel("Converged Mean Value")
    plt.title("converged Mean Value vs gamma")
    plt.legend(('fire possibility = 0.1', 'fire possibility = 0.9'), loc="upper left")
    plt.show()



if __name__ == '__main__':
    gamma = 0.9
    num_states = 20
    r1 = 4
    r2 = 2
    p_fire = 0.1
    P,R = forest(num_states, r1, r2, p_fire)
    vi = ValueIteration(P, R, 0.96, 1e-20)
    vi.run()

    P2, R2 = forest(num_states, r1, r2, 0.8)
    vi2 = ValueIteration(P2, R2, 0.96, 1e-20)
    vi2.run()

    # # calculate and plot the v_mean
    # iter_score(vi, vi2)

    # gamma_iter_value()
    # #
    #

    pi = PolicyIteration(P, R, 0.96)
    pi.run()

    pi2 = PolicyIteration(P2, R2, 0.96)
    pi2.run()
    # iter_score(pi, pi2)
    # #iter_policy(pi, pi2)
    # gamma_iter_value_p()

    q = QLearning(P,R,0.4, alpha=0.9,n_iter=100000)
    q.run()

    q2 = QLearning(P2, R2,0.4, alpha=0.9, n_iter=100000)
    q2.run()
    iter_score(q, q2)
