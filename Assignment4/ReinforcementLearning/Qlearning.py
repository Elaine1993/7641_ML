import gym
import numpy as np
from gym import wrappers
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt

import gym
import numpy as np

def q_learning(gma, is_slippery):
    # 1. Load Environment and Q-table structure
    env = gym.make('FrozenLake8x8-v0', is_slippery= is_slippery)
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    # env.obeservation.n, env.action_space.n gives number of states and action in env loaded
    # 2. Parameters of Q-leanring
    eta = .628
    count = 0
    epis = 5000
    rev_list = [] # rewards per episode calculate
    rev_list_true = []
    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        total_reward = 0
        while j < 99:
            # env.render()
            count +=1
            j+=1
            # Choose action from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        # env.render()
        reward_sum  = sum(rev_list)/epis
    # print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
    return reward_sum, count

if __name__ == '__main__':
    gamma_range = np.arange(0.1,1.0,0.1)
    reward_sum = []
    gamma = []
    rev_sum_deterministic = []
    rev_sum_stochastic = []

    count_deter = []
    count_stoc = []
    for i in gamma_range:
        print (i)
        gamma.append(i)

        reward_sum, count = q_learning(i, False)
        rev_sum_deterministic.append(reward_sum)
        count_deter.append(count)

        reward_sum, count = q_learning(i, True)
        rev_sum_stochastic.append(reward_sum)
        count_stoc.append(count)

    # plt.plot(gamma, count_deter,linestyle='--', marker='o', color='b')
    # plt.xlabel("gamma")
    # plt.ylabel("iteration")
    # plt.title("Deterministic: gamma vs converged iteration")
    # plt.show()
    #
    # #
    #
    # plt.plot(gamma, rev_sum_stochastic, linestyle='--', marker='o', color='b')
    # plt.xlabel("gamma")
    # plt.ylabel("total reward per iteration")
    # plt.title("Stochastic: gamma vs reward")
    # plt.show()
    #
    plt.plot(gamma, count_deter, marker = 'o')
    plt.plot(gamma, count_stoc, marker='o')
    plt.legend(('deterministic', 'stochastic'),loc= "upper left")
    plt.xlabel("Gamma")
    plt.ylabel("iteration")
    plt.title("gamma vs converged iteration")
    plt.show()
