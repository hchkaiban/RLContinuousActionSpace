#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:44:06 2018

@author: uidr8963
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt

pathP =r'/home/uidr8963/Dokumente/TempModel/RL_TempModel/RL_TempModel/'

StateIsAllObs = True

class Plant:
    
    Plant_MaxSteps = 500
    Plant_MaxAbsOutput = 30
    FullRandInputs = False
    
    def __init__(self):
        np.random.seed = 55
        self.steps = 0
        self.plant_o = []
        self.model_o = []
        self.Input1=[] 
        self.Input2=[] 
        self.Input3=[] 
        self.Input4=[] 
        self.Input5=[]
        self.Param1=[]; self.Param2=[]; self.r=[]
        if StateIsAllObs:
            self.observation_space = np.zeros(5)
        else:
            self.observation_space = np.zeros(2)
        self.action_space = np.zeros(2)
        self.action_range = [[6,11],[0,2]] #format [[fac, fac], [ofs, ofs]]
        self.done = False
    
    def sat(self, inp, minp=-2**32+1, maxp=2**32-1):
        return(min(max(inp, minp), maxp))
         
    fig, ax = plt.subplots(4,1)
    def plot_logs(self, f, ax, Pl_o, Ml_o, I1, I2, I3, I4, I5):
        f.set_figwidth(10)
        f.set_figheight(10)
        #ax1.set_xlim([0,50])
        ax[0].plot(Pl_o, label= 'Plant Out')
        ax[0].plot(Ml_o, label= 'Model Out')
        ax[0].set_title('Plant vs Model')
        ax[0].legend(loc='lower right')
        
        ax[1].plot(self.Param1, label= 'Param1')
        ax[1].plot(self.Param2, label= 'Param2')
        ax[1].legend(loc='lower right')
        #ax[1].set_title('Parameters')
        
        ax[2].plot(self.r, label= 'Episode reward')
        ax[2].legend(loc='lower right')
        #ax[2].set_title('Rewared')
        
        #ax2.set_xlim([0,50])
        ax[3].plot(I1, label='Input 1')
        ax[3].plot(I2, label='Input 2')
        ax[3].plot(I3, label='Input 3')
        ax[3].plot(I4, label='Input 4')
        ax[3].plot(I5, label='Input 5')
        ax[3].set_title('Plant Inputs')
        ax[3].legend(loc='lower right')
        
        f.savefig('Plant_DDQN_Render.png')
        ax[0].cla(); ax[1].cla(); ax[2].cla(); ax[3].cla()
        
        plt.close(f)
        #gc.collect()
    
    # Block 1
    x = np.arange(3,8)
    y = np.r_[0, 1,5,0,3]
    Ip1_blk1 = interp1d(x, y)
    #print(Ip1_blk1(5.5))
    
    xx = np.arange(0,6)
    yy = np.arange(0,3)
    zz = np.array([[-5,2,5,-2,3,5], [-2,-5,5,2,5,0], [5,-1,-2,0,4,2]])
    Ip2_blk1 = interp2d(xx, yy, zz)
    #print(Ip2_blk1(5.5, 0) [0])
    
    
    #Block 2
    c1_blk2 = 20
    c2_blk2 = 10
    xxx = np.arange(-5,2)
    yyy = np.r_[0,1,5,20,50,20,10]
    Ip1_blk2 = interp1d(xxx, yyy)
    #print(Ip1_blk2(-3))
    
    
    #Block 3
    c_blk3 = 9.2
    xxxx = np.r_[2,6,10,12,16,20,25,30]
    yyyy = np.r_[-120,-20,-2,-1,0,1,5,20]
    Ip1_blk3 = interp1d(xxxx, yyyy)
    #print(Ip1_blk3(2.4))
    
    # Build arbitrary plant
    def blk1(self, in1, in2, in3):
        in1 = self.sat(in1, 3, 7)
        ip1 = self.Ip1_blk1(in1)
        self.Param1.append(ip1)

        v1 = ip1 + in2
        v1 = self.sat(v1, 1, 5)
        in3 = self.sat(in3, 0, 2)
        ip2 = self.Ip2_blk1(v1, in3)
        #print ('blk1 plt ip2', ip2[0], end=' ')
        return ip2[0]
    
    def blk2(self, in1):
        in1 = self.sat(in1, -5, 1)
        ip1 = self.Ip1_blk2(in1)
        
        if ip1 > 10:
            return self.c1_blk2
        else:
            return self.c2_blk2
        
    def blk3(self, in1, in2):
        v1 = in2 + self.c_blk3
        self.Param2.append(self.c_blk3)
        #print ('blk3 plt in1', in1, end=' ')
        in1 = self.sat(in1, 2, 29)
        ip1 = self.Ip1_blk3(in1)
        return v1 * ip1
    
    def plant(self, in1, in2, in3, in4, in5):
        i_blk3 = self.blk1(in1, in2, in3) + self.blk2(in4)
        return(self.blk3(i_blk3, in5))
        

    # Build model to be optimized or environment in RL world
    def blk1_mdl(self, in1, in2, in3, action):
        in1 = self.sat(in1, 3, 7)
        #ip1 shall be identified
        #ip1 = self.Ip1_blk1(in1)
        ip1 = in1 * action
        
        v1 = ip1 + in2
        v1 = self.sat(v1, 1, 5)
        in3 = self.sat(in3, 0, 2)
        ip2 = self.Ip2_blk1(v1, in3)
        #print ('blk1 mdl ip2', ip2[0])
        return ip2[0]   
    
    def blk3_mdl(self, in1, in2, action):
        #c_blk3 shall be identified
        v1 = in2 + action
        in1 = self.sat(in1, 2, 29)
        #print ('blk3 mdl in1', in1)
        ip1 = self.Ip1_blk3(in1)
        return v1 * ip1
     
    def model(self, in1, in2, in3, in4, in5, action):
        i_blk3 = self.blk1_mdl(in1, in2, in3, action[0]) + self.blk2(in4)
        return(self.blk3_mdl(i_blk3, in5, action[1]))
        
    def reset(self):
        self.steps = 0
        self.plant_o = []; self.Input1=[]; self.Input2=[]; self.Input3=[]; self.Input4=[]; self.Input5=[]
        self.model_o = []
        self.Param1=[]; self.Param2=[]; self.r=[]
        if StateIsAllObs:
            self.observation_space = np.zeros(5)
        else:
            self.observation_space = np.zeros(2)
        self.action_space = np.zeros(2)
        self.done = False
        return self.observation_space
        
    def render(self):
        self.plot_logs(self.fig, self.ax, self.plant_o, self.model_o, self.Input1, self.Input2, self.Input3, self.Input4, self.Input5)
    
    def run(self):
        if self.steps < self.Plant_MaxSteps :
            if self.FullRandInputs:
                self.Input1.append = np.random.rand()*3 + 6     #check consistency with sat in related blk
                self.Input2.append = np.random.rand()*20 - 8    #check consistency with sat in related blk
                self.Input3.append = np.random.rand()*2         #check consistency with sat in related blk
                self.Input4.append = np.random.rand()*6 - 5     #check consistency with sat in related blk
                self.Input5.append = np.random.rand()*7 - 4     #check consistency with sat in related blk
            else:
                noise = np.random.rand()*0.08
                self.Input1.append(np.sin(self.steps*0.08) * 2 + 4 + np.sqrt(self.steps) / 10) 
                self.Input2.append(np.exp(-self.steps/self.Plant_MaxSteps) + np.exp(self.steps/self.Plant_MaxSteps) +noise -2) 
                self.Input3.append(np.tanh(self.steps*0.005) * 2 +noise)
                self.Input4.append(np.sin(self.steps*0.04) * 3 - 2)
                self.Input5.append(np.exp(-3*self.steps/self.Plant_MaxSteps) + np.sin(self.steps*0.04) +noise - 2)
            
#            if self.steps % 20 == 0:
#                print(str(self.steps) +'/'+str(self.Plant_MaxSteps), end=' ')
            
            self.plant_o.append(self.plant(self.Input1[-1], self.Input2[-1], self.Input3[-1], self.Input4[-1], self.Input5[-1]))
            return False
        else:
            return True
         
    def step(self, act):
        self.done = self.run()
        self.model_o.append(self.model(self.Input1[-1], self.Input2[-1], self.Input3[-1], self.Input4[-1], self.Input5[-1], act))
        self.steps += 1
        if StateIsAllObs:
            #State is vector of all the observations i.e. all the model inputs
            self.observation_space = np.array([self.Input1[-1], self.Input2[-1], self.Input3[-1], self.Input4[-1], self.Input5[-1]])
        else:
            #State is a vector of selected observations influencing the parameters to be identified
            self.observation_space = np.array([self.Input1[-1], self.Input5[-1]])
              
        #r = 1 - (np.abs(self.plant_o[-1] - self.model_o[-1]) / self.Plant_MaxAbsOutput)
        err = self.plant_o[-1] - self.model_o[-1]
        r = 1 - (np.square(err) / np.square(self.Plant_MaxAbsOutput))
        self.r.append(r)
#        if abs(err) < 1 :
#            r = 1 - (np.square(err))
#        else:
#            if abs(err) < 30:
#                r = 1 - (np.square(err) / np.square(30)) 
#            else:
#                r = 1 - (np.square(err) / np.square(self.Plant_MaxAbsOutput))
        #r = 1/ np.square(err)
        #r = np.clip(r, 0, 100)
        #r_ += r
        if self.done:
            pass
            #self.Param1=[]; self.Param2=[]; self.r=[]
        
        info = 'Arbitrary plant model for RL investigations'
        return self.observation_space, r, self.done, info 
    
    def close(self):
        plt.close(self.fig)
        self.reset()
        
if __name__ == "__main__":

    actions = [0.3,9.2]
    plant_ = Plant()
    
    while not plant_.done:
        s, r, done, _ = plant_.step(actions)    
        print(r)
  
    plant_.render()
    plant_.close()