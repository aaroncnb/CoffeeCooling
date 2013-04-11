# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:29:30 2013

@author: hok1
"""

import numpy as np
from scipy.integrate import odeint
import pylab as pl

class NewtonCoolingLaw:
    def __init__(self, roomTemp, coffeeInitTemp):
        self.roomTemp = roomTemp
        self.coffeeInitTemp = coffeeInitTemp
        self.coolingConst = 0.1
        self.creamerTempDrop = 5.0

    def getDecayRate(self, temp, time):
        return (-self.coolingConst * (temp-self.roomTemp))
        
    def solveODE(self, startTime, endTime, timeSteps=100):
        timeVec = np.linspace(startTime, endTime, timeSteps)
        tempVec = odeint(self.getDecayRate, self.coffeeInitTemp, timeVec)
        return timeVec, tempVec
        
    def coolingAfterCreamer(self, startTime, endTime, timeSteps=100):
        timeVec = np.linspace(startTime, endTime, timeSteps)
        tempVec = odeint(self.getDecayRate,
                         max(self.coffeeInitTemp-self.creamerTempDrop, 
                             self.roomTemp), 
                        timeVec)
        return timeVec, tempVec

    def coolingBeforeCreamer(self, startTime, endTime, timeSteps=100):
        timeVec = np.linspace(startTime, endTime, timeSteps)
        tempVec = odeint(self.getDecayRate, self.coffeeInitTemp, timeVec)
        tempVec[timeSteps-1] = max(tempVec[timeSteps-1] - self.creamerTempDrop, 
                                self.roomTemp)
        return timeVec, tempVec

    def coolingWithMultipleAddingCream(self, startTime, endTime, 
                                       timeAddingCreamVec, timeSteps=100):
        timeVec = np.linspace(startTime, endTime, timeSteps)
        numTimes = timeAddingCreamVec.size
        if numTimes==0:
            return self.solveODE(startTime, endTime, timeSteps)
        ndarrayPosAddingCream = np.zeros(numTimes)
        timeStep = float(endTime-startTime)/(timeSteps-1)
        for i in range(numTimes):
            ndarrayPosAddingCream[i]=int(np.floor(timeAddingCreamVec[i]/timeStep))
        ndarrayPosAddingCream.sort()
        eachCreamTempDrop = self.creamerTempDrop/numTimes
        startTemp = self.coffeeInitTemp
        timeStepStartPos = 0
        tempVec = np.array([])
        for i in range(numTimes):
            posAddingCream = ndarrayPosAddingCream[i]
            tempVec1 = odeint(self.getDecayRate, startTemp, 
                              timeVec[timeStepStartPos:(posAddingCream+1)])
            tempVec = np.append(tempVec, tempVec1)
            timeStepStartPos = posAddingCream + 1
            startTemp = tempVec1[-1] - eachCreamTempDrop
        tempVec1 = odeint(self.getDecayRate, startTemp,
                          timeVec[(ndarrayPosAddingCream[-1]+1):])
        tempVec = np.append(tempVec, tempVec1)
        return timeVec, tempVec
        
def main():
    ncl = NewtonCoolingLaw(25, 100)
    timeVec, tempVec = ncl.solveODE(0, 10)
    timeVec1, tempVec1 = ncl.coolingAfterCreamer(0, 10)
    timeVec2, tempVec2 = ncl.coolingBeforeCreamer(0, 10)
    timeVec3, tempVec3 = ncl.coolingWithMultipleAddingCream(0, 10,
                                                            np.array([3, 6, 9]))
    pl.plot(timeVec1, tempVec1)
    pl.plot(timeVec2, tempVec2)
    pl.plot(timeVec3, tempVec3)
    pl.title('Temperature of the Coffee over Time')
    pl.ylabel('Temperature (degrees Celcius)')
    pl.xlabel('Time')
    
if __name__ == '__main__':
    main()
