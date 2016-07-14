import numpy as np
from scipy.integrate import odeint
from functools import partial

coolingfcn = lambda temp, t0, k, roomtemp: -k*(temp-roomtemp)

cooling_parameters = {'init_temp': 94.0,      # initial temperature of the coffee (degrees Celcius)
                      'cube_dectemp': 5.0,    # temperature decrease due to the cube
                      'roomtemp': 25.0,       # room temperature
                      'k': 1.0/15.0           # cooling parameter (inverse degrees Celcius)
                      }

# adding sugar cube before cooling
def cube_before_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    y0 = init_temp - cube_dectemp
    temps = odeint(partial(coolingfcn, k=k, roomtemp=roomtemp), y0, t)
    return temps

# adding sugar cube after cooling
def cube_after_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    temps = odeint(partial(coolingfcn, k=k, roomtemp=roomtemp), init_temp, t)
    temps[-1] -= cube_dectemp
    return temps

# adding sugar continuously
def continuous_sugar_cooling(t, init_temp, cube_dectemp, roomtemp, k):
    timespan = t[-1] - t[0]
    newcoolingfcn = lambda temp, t0: coolingfcn(temp, t0, k, roomtemp) - cube_dectemp / timespan
    temps = odeint(newcoolingfcn, init_temp, t)
    return temps

# adding sugar at a specified time(s)
def sugar_specifiedtime_cooling(t, sugar_time_indices, init_temp, cube_dectemp, roomtemp, k):
    sorted_sugar_time_indices = np.sort(sugar_time_indices)
    num_portions = len(sorted_sugar_time_indices)

    temps = np.array([])
    temp = init_temp
    t_segments = np.split(t, sorted_sugar_time_indices)
    num_segments = len(t_segments)
    for i in range(num_segments):
        temp_segment = cube_after_cooling(t_segments[i],
                                          temp,
                                          float(cube_dectemp)/num_portions,
                                          roomtemp,
                                          k)
        temps = np.append(temps, temp_segment)
        temp = temp_segment[-1]
    temps[-1] += float(cube_dectemp)/num_portions

    return temps