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

def sugar_specifiedtime_cooling(t, sugar_time_indices, init_temp, cube_dectemp, roomtemp, k):
    sorted_sugar_time_indices = np.sort(sugar_time_indices)
    num_portions = len(sorted_sugar_time_indices)

    temps = np.array([])
    temp = init_temp
    t_segments = np.split(t, sorted_sugar_time_indices)
    num_segments = len(t_segments)
    for i in range(num_segments):
        temp_segment = cube_after_cooling(t_segments[i], temp, float(cube_dectemp)/num_portions, roomtemp, k)
        temps = np.append(temps, temp_segment)
        temp = temp_segment[-1]
    temps[-1] += float(cube_dectemp)/num_portions

    return temps