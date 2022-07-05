# -*- coding: utf-8 -*-
"""
Created on Sun May 15 22:37:00 2016

@author: Hossam Faris
"""

import random
import numpy
import time
import bm

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual = []
        self.convergence = []
        self.optimizer = ""
        self.objfname = ""
        self.startTime = 0
        self.endTime = 0
        self.executionTime = 0
        self.lb = 0
        self.ub = 0
        self.dim = 0
        self.popnum = 0
        self.maxiers = 0

def PSO(objf, lb, ub, dim, PopSize, iters):

    # PSO parameters

    Vmax = 6
    wMax = 0.9
    wMin = 0.2
    c1 = 2
    c2 = 2

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    vel = numpy.zeros((PopSize, dim))

    pBestScore = numpy.zeros(PopSize)
    pBestScore.fill(float("inf"))

    pBest = numpy.zeros((PopSize, dim))
    gBest = numpy.zeros(dim)

    gBestScore = float("inf")

    pos = numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(iters)

    ############################################
    #print('PSO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, PopSize):
            # pos[i,:]=checkBounds(pos[i,:],lb,ub)
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])
            # Calculate objective function for each particle
            fitness = objf(pos[i, :])

            if pBestScore[i] > fitness:
                pBestScore[i] = fitness
                pBest[i, :] = pos[i, :].copy()

            if gBestScore > fitness:
                gBestScore = fitness
                gBest = pos[i, :].copy()

        # Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iters)

        for i in range(0, PopSize):
            for j in range(0, dim):
                r1 = random.random()
                r2 = random.random()
                vel[i, j] = (
                    w * vel[i, j]
                    + c1 * r1 * (pBest[i, j] - pos[i, j])
                    + c2 * r2 * (gBest[j] - pos[i, j])
                )

                if vel[i, j] > Vmax:
                    vel[i, j] = Vmax

                if vel[i, j] < -Vmax:
                    vel[i, j] = -Vmax

                pos[i, j] = pos[i, j] + vel[i, j]

        convergence_curve[l] = gBestScore

        #if l % 1 == 0:
        #    print(
        #        [
        #            "At iteration "
        #            + str(l + 1)
        #            + " the best fitness is "
        #            + str(gBestScore)
        #        ]
        #    )
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "PSO"
    s.objfname = objf.__name__

    return s
if __name__ == '__main__':
    dim=30
    popSize=50
    Iter=500
    z_a=[]
    z=0
    t=0
    t_a=[]
    n=10
    lb=[-100,-10,-100,-100,-30,-100,-1.28,-500,-5.12,-32,-600,-50,-50]
    ub=[100, 10, 100, 100, 30, 100, 1.28, 500, 5.12, 32, 600, 50, 50]
    #s=PSO(getattr(bm, "F1"), lb, ub, dim, popSize, Iter)
    for i in range(13):
        print('PSO is now tackling  "' +"F"+str(i+1)+ '"')
        for j in range(n):
            s=PSO(getattr(bm, "F"+str(i+1)), lb[i], ub[i], dim, popSize, Iter)
            z+=s.convergence[-1]
            t+=s.executionTime
        z_a.append((z/n))
        t_a.append((t/n))
        z=0
        t=0
        print("F"+str(i+1)+"  "+str(z_a[i]))
        print("时间 " + str(t_a[i]) + " s")
    asq = numpy.array(z_a)
    asb =numpy.array(t_a)
    numpy.savetxt("D:\\data\\PSOz.csv", asq, fmt='%f', delimiter=",")
    numpy.savetxt("D:\\data\\PSOt.csv", asb, fmt='%f', delimiter=",")
