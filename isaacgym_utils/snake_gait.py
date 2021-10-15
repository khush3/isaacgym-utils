"""
Source: https://github.com/biorobotics/SnakeLib
"""

import numpy as np

class Gait:

    def __init__(self, numModules):
        self.numModules = numModules
        self.selectedGait = 'LateralUndulation1'

    def setGait(self,selectGait):
        self.selectedGait = selectGait

    def getSignal(self,t):
        if self.selectedGait == 'LateralUndulation1':
            signal = self.lateralUndulation1(t)
        elif self.selectedGait == 'LateralUndulation2':
            signal = self.lateralUndulation2(t)
        elif self.selectedGait == 'sidewindingRSS2020':
            signal = self.sidewindingRSS2020(t)
        elif self.selectedGait == 'sidewinding':
            signal = self.sidewinding(t)
        elif self.selectedGait == 'rolling':
            signal = self.rolling(t)
        elif self.selectedGait == 'slithering':
            signal = self.slithering(t)
        elif self.selectedGait == 'turninginplace':
            signal = self.turninginplace(t)
        elif self.selectedGait == 'rolling_helix':
            signal = self.rolling_helix(t)
        else:
            signal = self.lateralUndulation1(t)
        
        return signal

    # Sidewinding (Baxi & Tianyu RSS 2020 paper).
    # Taken from: sidewinding_test.py
    def sidewindingRSS2020(self,t):
        numWaves 		  = 1.5
        spFreq_odd        = 2 * np.pi * numWaves
        spFreq_even       = 1.0 * spFreq_odd
        amp_odd           = np.pi/180*50
        amp_even          = np.pi/180*75
        tmFreq            = 1
        phi0_even         = np.pi/2

        angles = np.zeros([1, self.numModules])
        for i in range(self.numModules):
            if i%2 == 1:
                angles[0, i] = amp_odd * np.sin(spFreq_odd * (i-1)/self.numModules - tmFreq * t)
            else:
                c_i = -1/(1+np.exp(-4 * np.sin(spFreq_even * (i)/self.numModules - tmFreq * t + phi0_even)))
                c_i1 = -1/(1+np.exp(-4 * np.sin(spFreq_even * (i+2)/self.numModules - tmFreq * t + phi0_even)))
                angles[0, i] = -amp_even * (c_i-c_i1)
        angles[0, 0] = -np.pi/2
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        angles = np.fliplr(angles)
        angles = np.multiply(reversals, angles)
        signal = []
        for i in range(self.numModules):
            signal.append(angles[0,i])

        return signal

    # Sidewinding (Traditional)
    # Taken from: https://github.com/biorobotics/reusnake/blob/master/demo_reusnake_python.py
    def sidewinding(self,t):
        
        # sidewinding parameters
        s = 0.5
        w = 1
        A_even = .6
        A_odd = .6
        delta = np.pi/4 # right
        # delta = -pi/4; # left
        beta_odd = 0
        beta_even = 0

        angles = np.zeros([1, self.numModules])
        for n in range(self.numModules):
            if n%2 == 1:
                angles[0, n] = beta_odd+A_odd*np.sin(n*s-w*t+delta)
            else:
                angles[0, n] = beta_even+A_even*np.sin(n*s-w*t)
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        angles = np.fliplr(angles)
        angles = np.multiply(reversals, angles)
        signal = []
        for i in range(self.numModules):
            signal.append(angles[0,i])

        return signal

    # Lateral Undulation (Traditional)
    # Written using: http://www.cs.cmu.edu/~mtesch/publications/Advanced_Robotics_2009.pdf
    # No need for angle reversals because motion is only in one direction
    def lateralUndulation1(self,t):
        A_odd = 0.75
        A_even = 0
        beta_odd = 0
        beta_even = 0.25  # for stability
        s = 4
        w = 2

        angles = np.zeros([1, self.numModules])
        for n in range(self.numModules):
            if n%2 == 1:
                angles[0, n] = beta_odd+A_odd*np.sin(n*s+w*t)
            else:
                angles[0, n] = beta_even+A_even*np.sin(n*s+w*t)
        # reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        # angles = np.fliplr(angles)
        # angles = np.multiply(reversals, angles)
        signal = []
        for i in range(self.numModules):
            signal.append(angles[0,i])
        
        return signal


    # Lateral Undulation
    # Taken from: snake_gait_test.py
    def lateralUndulation2(self,t):
        A = np.pi/6
        s = 4
        w = 2
        signal = []
        for n in range(self.numModules):
            if n%2 == 1:
                signal.append(-A*np.sin(n*s+w*t))
            else:
                signal.append(0)
        return signal

    def rolling(self,t):
        
        # rolling parameters
        wS = 0
        wT= 2
        A_even =0.1
        A_odd =0.1
        delta = np.pi/2
        beta_odd = 0
        beta_even = 0


        J_angles = np.zeros([1, self.numModules])
        for n in range(self.numModules):
            if n%2 == 1:
                J_angles[0, n] = beta_odd+A_odd*np.sin(wS*n-wT*t+delta)
            else:
                J_angles[0, n] = beta_even+A_even*np.sin(wS*n-wT*t)
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        J_angles = np.fliplr(J_angles)
        J_angles = np.multiply(reversals, J_angles)
        signal = []
        for i in range(self.numModules):
            signal.append(J_angles[0,i])

        return signal

    # needs tuning
    def slithering(self,t):
        
        # slithering parameters
        wS = 1
        wS1 = 2
        wT= 1
        wT1= 2
        A_even = 1
        A_odd = 0.3
        delta = np.pi/4
        beta_odd = 0
        beta_even =0.1


        J_angles = np.zeros([1, self.numModules])
        for x in range(self.numModules):
            if x%2 == 1:
                J_angles[0, x] = beta_odd + A_odd*np.sin(wS*x - wT*t + delta)
            else:
                J_angles[0, x] = beta_even + A_even*np.sin(wS1*x - wT1*t)
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        J_angles = np.fliplr(J_angles)
        J_angles = np.multiply(reversals, J_angles)
        signal = []
        for i in range(self.numModules):
            signal.append(J_angles[0,i])

        return signal

    def rolling_helix(self,t):
    
        # rolling parameters
        wS = 3.1 #Theta with respect to module number (spatial frequency)(vertical)
        wT= 1 #dTheta with respect to time
        A_even = 1.2
        A_odd = 1.2
        delta = np.pi/2
        beta_odd = 0
        beta_even = 0

        J_angles = np.zeros([1, self.numModules])
        for n in range(self.numModules):
            if n%2 == 1:
                J_angles[0, n] = beta_odd+A_odd*np.sin(wS*n-wT*t+delta)
            else:
                J_angles[0, n] = beta_even+A_even*np.sin(wS*n-wT*t)
        reversals = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
        J_angles = np.fliplr(J_angles)
        J_angles = np.multiply(reversals, J_angles)
        signal = []
        for i in range(self.numModules):
            signal.append(J_angles[0,i])

        return signal