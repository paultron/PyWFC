from enum import Enum
import random
import math
import numpy as np
from typing import Optional
from helper import Helper

class Model:
    _wave : Optional[np.ndarray] # 2d

    _propagator : np.ndarray # 3d
    compatible : np.ndarray # 3d
    _observed : list[int] # 1d

    stack : list[tuple[int, int]] # (int, int)
    stacksize : int
    observedSoFar : int

    _MX: int
    _MY : int
    _MXY : int
    _T : int
    _N : int

    _periodic : bool
    _ground : bool

    _weights : list[float]
    weightLogWeights : list[float]
    distribution : list[float]

    _sumsOfOnes : list[int]
    sumOfWeights : float
    sumOfWeightLogWeights : float
    startingEntropy : float

    _sumsOfWeights : list[float]
    _sumsOfWeightLogWeights : list[float]
    _entropies : list[float]

    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    opposite = [2,3,0,1]

    Heuristic = Enum('Heuristic', ['Entropy', 'MRV', 'Scanline'])

    heuristic : Heuristic

    def __init__(self, width:int, height:int, N, periodic, heuristic):
        self._MX = width
        self._MY = height
        self._MXY = width * height
        self._N = N
        self._periodic = periodic
        self.heuristic = heuristic

        # ... other member variables

    def Init(self):
        # ... initialization code
        self._wave = np.empty([self._MXY, self._T])
        self.compatible = np.empty([len(self._wave), self._T, 4])

        #for i in range(len(self._wave)):
            #self._wave[i] = np.empty(self._T, dtype=bool)
            #self.compatible[i] = np.empty(self._T)
            #for t in range(self._T):
                #self.compatible[i][t] = np.empty(4)

        self.distribution = np.empty(self._T)
        self._observed = np.empty(self._MXY)

        self.weightLogWeights = np.empty(self._T)
        self.sumOfWeights = 0
        self.sumOfWeightLogWeights = 0

        for t in range(self._T):
            self.weightLogWeights[t] = self._weights[t] * np.log(self._weights[t])
            self.sumOfWeights += self._weights[t]
            self.sumOfWeightLogWeights += self.weightLogWeights[t]

        self.startingEntropy = np.log(self.sumOfWeights) - self.sumOfWeightLogWeights / self.sumOfWeights

        self._sumsOfOnes = np.empty(self._MXY)
        self._sumsOfWeights = np.empty(self._MXY)
        self._sumsOfWeightLogWeights = np.empty(self._MXY)
        self._entropies = np.empty(self._MXY)

        self.stack = np.empty(len(self._wave * self._T), dtype=tuple) #new (int, int)[wave.Length * T]
        self.stacksize = 0

    def Run(self, seed, limit) -> bool:
        # ... run method
        if self._wave is None:
            self.Init()

        self.Clear()
        _random: random.Random = random.Random(seed)
        _l = 0
        while (_l < limit or limit < 0):
            _l += 1
            node: int = self.NextUnobservedNode(_random)
            if (node >= 0):
                self.Observe(node, _random)
                success: bool = self.Propagate()
                if not success: return False
            else:
                for i in range(len(self._wave)):
                    for t in range(self._T):
                        if (self._wave[i][t]):
                            self._observed[i] = t
                            break
                return True
        return True

    def NextUnobservedNode(self, random: random.Random):
        # ... next unobserved node logic
        if (self.heuristic == self.Heuristic.Scanline):
            _i = self.observedSoFar
            while _i < len(self._wave):
                if ((not self._periodic) and 
                    ((_i % self._MX + self._N > self._MX) 
                     or (_i / self._MX + self._N > self._MY))):
                        _i += 1
                        continue
                if (self._sumsOfOnes[i] > 1):
                    self.observedSoFar = _i + 1
                    return _i
                    
                _i += 1
            return -1
        
        min = 1e4
        argmin = -1
        for i in range(len(self._wave)):
            if (not self._periodic and 
                    ((_i % self._MX + self._N > self._MX) 
                     or (_i / self._MX + self._N > self._MY))):
                        _i += 1
                        continue
            remainingValues = self._sumsOfOnes[i]
            entropy = self._entropies[i] if self.heuristic == self.Heuristic.Entropy else remainingValues

            if (remainingValues > 1 and entropy <= min):
                noise = 1e-6 * random.random()
                if (entropy + noise < min):
                    min = entropy + noise
                    argmin = i
        return argmin

    def Observe(self, node: int, random: random.Random):
        # ... observe method
        w = self._wave[node]
        for t in range(self._T):
            self.distribution[t] = self._weights[t] if w[t] else 0.0
        r: int = Helper.RandomHelper(self.distribution, random.random())
        for t in range(self._T):
            if (w[t] != (t == r)):
                self.Ban(node, t)
    
    def Propagate(self):
        # ... propagate method
        while (self.stacksize > 0):
            i1, t1 = self.stack[self.stacksize - 1]
            self.stacksize -= 1

            x1 = i1 % self._MX
            y1 = i1 / self._MX

            for d in range(4):

                x2 = x1 + self.dx[d]
                y2 = y1 + self.dy[d]
                if ((not self._periodic) and (x2 < 0 or y2 < 0 
                                              or x2 + self._N > self._MX 
                                              or y2 + self._N > self._MY)):
                    continue

                if (x2 < 0): x2 += self._MX
                elif (x2 >= self._MX): x2 -= self._MX
                if (y2 < 0): y2 += self._MY
                elif (y2 >= self._MY): y2 -= self._MY

                i2 = x2 + y2 * self._MX
                p = self._propagator[d][t1]
                compat = self.compatible[i2]

                for l in range(len(p)):
                    t2 = p[l]
                    comp = compat[t2]

                    comp[d] -= 1
                    if (comp[d] == 0): self.Ban(i2, t2)
            
        return self._sumsOfOnes[0] > 0


    def Ban(self, i, t):
        # ... ban method
        self._wave[i][t] = False

        comp = self.compatible[i][t]
        for d in range(4): comp[d] = 0
        #for (int d = 0; d < 4; d++) comp[d] = 0
        self.stack[self.stacksize] = (i, t)
        self.stacksize += 1

        self._sumsOfOnes[i] -= 1
        self._sumsOfWeights[i] -= self._weights[t]
        self._sumsOfWeightLogWeights[i] -= self.weightLogWeights[t]

        sum = self._sumsOfWeights[i]
        self._entropies[i] = math.log(sum) - self._sumsOfWeightLogWeights[i] / sum

    def Clear(self):
        # ... clear method
        for i in range(len(self._wave)):
            for t in range(self._T):
                self._wave[i][t] = True
                for d in range(4):
                    self.compatible[i][t][d] = len(self._propagator[self.opposite[d]][t])
            self._sumsOfOnes[i] = len(self._weights)
            self._sumsOfWeights[i] = self.sumOfWeights
            self._sumsOfWeightLogWeights[i] = self.sumOfWeightLogWeights
            self._entropies[i] = self.startingEntropy
            self._observed[i] = -1
        self.observedSoFar = 0

        if (self._ground):
            for x in range(self._MX):
                for t in range(self._T - 1): self.Ban(x + (self._MY - 1) * self._MX, t)
                for y in range(self._MY - 1): self.Ban(x + y * self._MX, self._T - 1)
            self.Propagate()

    def Save(self, filename):
        # ... save method
        pass