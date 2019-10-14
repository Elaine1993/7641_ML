
import numpy as np
from random import random
import mlrose
import matplotlib.pyplot as plt
from datetime import datetime


def four_Peak():
    problem =mlrose.DiscreteOpt(length = 20, fitness_fn = mlrose.FourPeaks(t_pct=0.1), maximize = True, max_val = 2)
    init_state = np.array([0]*20)
    startTime = datetime.now()
    best_state, best_fitness, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=1000, max_iters=2500, restarts=0,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    rhcTime = totalTime.total_seconds()
    print("RHC")
    print("Time: ", rhcTime)
    print("best_state: " , best_state)
    print("best_fitness: ", best_fitness)
    print("Iteration: %d " % len(fitness_curve_rhc))


    ###############################
    startTime = datetime.now()
    best_statesa, best_fitnesssa, fitness_curve_sa = mlrose.simulated_annealing(problem, max_attempts=1000, max_iters=2500,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    saTime = totalTime.total_seconds()
    print("SA")
    print("Time: ", saTime)
    print("best_state: " , best_statesa)
    print("best_fitness: ", best_fitnesssa)
    print("Iteration: %d " % len(fitness_curve_sa))

    ###############################
    startTime = datetime.now()
    best_statega, best_fitnessga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts=1000, max_iters=2500,
                     curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    gaTime = totalTime.total_seconds()
    print("GA")
    print("Time: ", gaTime)
    print("best_state: ", best_statega)
    print("best_fitness: ", best_fitnessga)
    print("Iteration:  %d " % len(fitness_curve_ga))

    ###############################
    startTime = datetime.now()
    best_statemm, best_fitnessmm, fitness_curve_mm = mlrose.mimic(problem, max_attempts=1000, max_iters=2500,
                      curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    mmTime = totalTime.total_seconds()
    print("MIMIC")
    print("Time: ", mmTime)
    print("best_state: ", best_statemm)
    print("best_fitness: ", best_fitnessmm)
    print("Iteration: %d " % len( fitness_curve_mm))



def one_max():
    problem =mlrose.DiscreteOpt(length = 20, fitness_fn = mlrose.OneMax(), maximize = True, max_val = 2)
    init_state = np.array([0]*20)
    startTime = datetime.now()
    best_state, best_fitness, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=1000, max_iters=2500, restarts=0,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    rhcTime = totalTime.total_seconds()
    print("RHC")
    print("Time: ", rhcTime)
    print("best_fitness: ", best_fitness)
    print("Iteration: %d " % len(fitness_curve_rhc))


    ###############################
    startTime = datetime.now()
    best_statesa, best_fitnesssa, fitness_curve_sa = mlrose.simulated_annealing(problem, max_attempts=1000, max_iters=2500,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    saTime = totalTime.total_seconds()
    print("SA")
    print("Time: ", saTime)
    print("best_fitness: ", best_fitnesssa)
    print("Iteration: %d " % len(fitness_curve_sa))

    ###############################
    startTime = datetime.now()
    best_statega, best_fitnessga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts=1000, max_iters=2500,
                     curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    gaTime = totalTime.total_seconds()
    print("GA")
    print("Time: ", gaTime)
    print("best_fitness: ", best_fitnessga)
    print("Iteration:  %d " % len(fitness_curve_ga))

    ###############################
    startTime = datetime.now()
    best_statemm, best_fitnessmm, fitness_curve_mm = mlrose.mimic(problem, max_attempts=1000, max_iters=2500,
                      curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    mmTime = totalTime.total_seconds()
    print("MIMIC")
    print("Time: ", mmTime)
    print("best_fitness: ", best_fitnessmm)
    print("Iteration: %d " % len( fitness_curve_mm))



def k_COLOR():
    problem =mlrose.DiscreteOpt(length = 20, fitness_fn = mlrose.MaxKColor(edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
), maximize = True, max_val = 2)
    init_state = np.array([0]*20)
    startTime = datetime.now()
    best_state, best_fitness, fitness_curve_rhc = mlrose.random_hill_climb(problem, max_attempts=1000, max_iters=2500, restarts=0,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    rhcTime = totalTime.total_seconds()
    print("RHC")
    print("Time: ", rhcTime)
    print("best_fitness: ", best_fitness)
    print("Iteration: %d " % len(fitness_curve_rhc))


    ###############################
    startTime = datetime.now()
    best_statesa, best_fitnesssa, fitness_curve_sa = mlrose.simulated_annealing(problem, max_attempts=1000, max_iters=2500,
                      init_state=init_state, curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    saTime = totalTime.total_seconds()
    print("SA")
    print("Time: ", saTime)
    print("best_fitness: ", best_fitnesssa)
    print("Iteration: %d " % len(fitness_curve_sa))

    ###############################
    startTime = datetime.now()
    best_statega, best_fitnessga, fitness_curve_ga = mlrose.genetic_alg(problem, max_attempts=1000, max_iters=2500,
                     curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    gaTime = totalTime.total_seconds()
    print("GA")
    print("Time: ", gaTime)
    print("best_fitness: ", best_fitnessga)
    print("Iteration:  %d " % len(fitness_curve_ga))

    ###############################
    startTime = datetime.now()
    best_statemm, best_fitnessmm, fitness_curve_mm = mlrose.mimic(problem, max_attempts=1000, max_iters=2500,
                      curve=True, random_state=1,
                      state_fitness_callback=None, callback_user_info=None)

    totalTime = datetime.now() - startTime
    mmTime = totalTime.total_seconds()
    print("MIMIC")
    print("Time: ", mmTime)
    print("best_fitness: ", best_fitnessmm)
    print("Iteration: %d " % len( fitness_curve_mm))



four_Peak()
one_max()
