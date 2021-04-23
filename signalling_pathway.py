from __future__ import division
import random
import numpy as np
import matplotlib.pyplot as plt


def gillespie(stoichiometry, rate, initial_conditions, max_time):
    """Function for the Gillespie algorithm.

    :param stoichiometry: stoichiometry matrix.
    :param rate: rate function that gives reaction propensities as a function of the current state.
    :param initial_conditions: number of molecules set at the beginning of the algorithm.
    :param max_time: maximum time of the simulation (in minutes).
    
    :return: next reaction and time.
    """

    time = 0.0  # initial starting time
    time_increment = [0.0]  # saves time increment from each iteration
    y = np.copy(initial_conditions)
    results = [np.array(list(y))]  # save result from each increment

    while True:  # continuous loop until break
        propensities = rate(y)
        a0 = sum(propensities) # total propensity
        if a0 == 0:  
            break
        time_interval = np.random.exponential(scale=(1.0/a0))  # exponentially generated time interval

        if time + np.all(time_interval) > max_time:  # break when maximum time is reached
            break
            
        next_reaction = random.choices(population=range(len(propensities)),  # next randomly chosen reaction
                                       weights=propensities,
                                       k=1)
        reaction_stoichiometry = stoichiometry[0:, next_reaction]  # change in molecule numbers from chosen reaction 
        reaction_stoichiometry.shape = len(reaction_stoichiometry)

        time += time_interval  # time increase with every iteration
        y += reaction_stoichiometry  # how the state of the propensities changes with each iteration

        # saving the time and results
        time_increment.append(time)
        results.append(list(y))
    return time_increment, np.array(results)

####### SIGNALING NETWORK #######

# DNA --> ptch_mrna
# ptch_mrna --> ptch(cytoplasm)
# ptch_mrna --> 0
# ptch(cytoplasm) --> ptch(membrane)
# ptch(membrane) --> 0
# shh + ptch(membrane) --> shh-ptch
# shh-ptch --> 0
# chol(o) --> chol(i)
# chol(i) + ptch(mem) --> chol(o) + ptch(mem)
# DNA --> smo
# smo --> 0
# chol(i) + smo --> smo-chol
# smo-chol --> chol(i) + smo
# smo-chol --> 0 
# DNA --> gli_fl
# gli_fl + smo-chol --> gli_a + smo-chol
# gli_fl --> 0
# gli_a --> 0
# gli_fl --> gli_r
# gli_r --> 0
# DNA + gli_a --> ptch_mrna + gli_a

stoichiometry = np.array([[1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  #ptch_mrna 0
                          [0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #ptch_cyt 1
                          [0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #ptch_mem 2
                          [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #shh 3
                          [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #shh-ptch 4
                          [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #chol-outer 5
                          [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #chol-inner 6
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #smo 7
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0],  #smo-chol 8
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, -1, 0, 0],  #gli_fl 9
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0],  #gli_a 10
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0]])  #gli_r 11
                          

def rates(y):
    return np.array([40.6485,  # 0 --> ptch_mrna
                     0.00626*y[0],  # ptch_mrna --> ptch_cyt
                     0.0058*y[0],  # ptch_mrna --> 0
                     0.0036*y[1],  # ptch_cyt --> ptch_mem
                     0.0068*y[2],  # ptch_mem --> 0
                     0.0000143*y[2]*y[3],  # shh + ptch_mem --> shh_ptch
                     0.0068*y[4],  # shh_ptch --> 0
                     0.009*y[5],  # chol(o) --> chol(i)
                     0.25*y[2]*y[6],  # chol(i) + ptch_mem --> chol(o) + ptch_mem
                     40.6485,  # 0 --> smo
                     0.0058*y[7],  # smo --> 0
                     0.0001072*y[6]*y[7],  # smo + chol(i) --> smo-chol
                     0.004*y[8],  # smo-chol --> smo + chol_i
                     0.0019*y[8],  # smo-chol --> 0
                     45.165,  # 0 --> gli_fl
                     0.00008701*y[8]*y[9],  # gli_fl + smo-chol --> gli_a + smo-chol
                     0.0029*y[9],  # gli_fl --> 0
                     0.0173*y[10],  # gli_a --> 0
                     0.0015*y[9],  # gli_fl --> gli_r
                     0.005*y[11],  # gli_r --> 0 
                     0.05*y[10]])  # mrna upreg


rate = lambda y: rates(y)
max_time = 500

### input shh = 20nm ###
inital_conditions = np.array([0, 3010, 301, 6020, 0, 12040, 12040, 0, 0, 0, 0, 2500])  

(time_increment, results) = gillespie(stoichiometry, rate, inital_conditions, max_time)

gli_a = results[:, 10]
gli_r = results[:, 11]


### input shh = 50nm ###
initial_conditions2 = np.array([0, 3010, 301, 15050, 0, 12040, 12040, 0, 0, 0, 0, 2500])

(time_increment2, results2) = gillespie(stoichiometry, rate, initial_conditions2 , max_time)

gli_a2 = results2[:, 10]
gli_r2 = results2[:, 11]


### input shh = 100nm ###
initial_conditions3 = np.array([0, 3010, 301, 30100, 0, 12040, 12040, 0, 0, 0, 0, 2500])

(time_increment3, results3) = gillespie(stoichiometry, rate, initial_conditions3, max_time)
gli_a3 = results3[:, 10]
gli_r3 = results3[:, 11]


### plot results ###
plt.plot(time_increment, gli_a, label="GliA - Shh = 20nm")
plt.plot(time_increment2, gli_a2, label="GliA - Shh = 50nm")
plt.plot(time_increment3, gli_a3, label="GliA - Shh = 100nm")

plt.plot(time_increment, gli_r, label="GliR - Shh = 20nm")
plt.plot(time_increment2, gli_r2, label="GliR - Shh = 50nm")
plt.plot(time_increment3, gli_r3, label="GliR - Shh = 100nm")

plt.xlabel("time (minutes)")
plt.ylabel("number of molecules")
plt.title("Gli activity in response to Shh signalling")
plt.grid()
plt.legend(loc=2)
plt.show()
