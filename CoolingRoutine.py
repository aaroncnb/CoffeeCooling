import numpy as np
from scipy.integrate import odeint
from functools import partial

coolingfcn = lambda temp, t0, k, roomtemp: -k*(temp-roomtemp)

cooling_parameter = {'init_temp': 94.0,
                     'cube_dectemp': 5.0,
                     'roomtemp': 25.0,
                     'k': 2.0
                     }

def cube_before_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    y0 = init_temp - cube_dectemp
    temps = odeint(partial(coolingfcn, k=k, roomtemp=roomtemp), y0, t)
    return temps

def cube_after_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    temps = odeint(partial(coolingfcn, k=k, roomtemp=roomtemp), init_temp, t)
    temps[-1] -= cube_dectemp
    return temps

def continuous_sugar_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    timespan = t[-1] - t[0]
    newcoolingfcn = lambda temp, t0, k, roomtemp: coolingfcn(temp, t0, k, roomtemp) - cube_dectemp / timespan
    temps = odeint(partial(newcoolingfcn, k=k, roomtemp=roomtemp), init_temp, t)
    return temps

def sugar_specifiedtime_cooling(t, sugar_times, init_temp, cube_dectemp, roomtemp, k):
    t0 = t[0]
    tend = t[-1]
    sorted_sugar_times = np.sort(filter(lambda t: t>=t0 and t<=tend, sugar_times))
    num_segments = len(sorted_sugar_times) + 1

    t_segments = np.split(t, sorted_sugar_times)
    # not finished