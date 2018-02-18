"""
Double Deep Policy Gradient with continuous action space, 
Reinforcement Learning.
Based on yapanlau.github.io and the Continuous control with deep reinforcement
learning ICLR paper from 2016

"""

from Env_Plant import Plant
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math, random

from ReplayBuffer import ReplayBuffer
from Models import ActorNetwork
from Models import CriticNetwork
from OU import OrnsteinUhlenbeckActionNoise
import TempConfig

LoadWeithsAndTrain = False
LoadWeithsAndTest = True

BUFFER_SIZE = 150000
BATCH_SIZE = 99
STATE_NORM = (10, 1) #(1/gain, ofs)
STATE_NORM_single = [(10, 0), (1, 0), (2, 0), (5, 0.9), (3, 1)] 
REW_MIN = -20

GAMMA = 0.99
TAU = 0.001     #Target Network HyperParameters

LRA = 0.0001    #Learning rate for Actor
LRC = 0.001     #Lerning rate for Critic

episode_count = 20000
max_steps = 150000

MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = int(max_steps)        # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay fn of episodes of learning agent

def plot_logs(f, ax1, reward, act1, act2, loss, eps):
    
    f.set_figwidth(10)
    f.set_figheight(10)

    ax1[0].set_title('Reward')
    #ax1.set_xlim([0,50])
    ax1[0].plot(reward, label= 'Reward')
    if len(reward) > 10:
        mav = np.convolve(reward, np.ones((10,))/10, mode='valid')
        ax1[0].plot(mav, label= 'MovingAv10')
    ax1[0].legend(loc='upper left')
   
    ax1[1].plot(act1, label= 'action1')  
    ax1[1].plot(act2, label= 'action2')
    ax1[1].legend(loc='upper right')
    ax1[3].plot(loss, label= 'Training loss')    
    ax1[3].legend(loc='upper right')

    ax1[2].plot(eps, label= 'Epsilon')    
    ax1[2].legend(loc='upper right')
    
    f.savefig('RL_DDPG_Plant.png')
    ax[0].cla(); ax[1].cla(); ax[2].cla(); ax[3].cla()
    #ax1.cla()
    plt.close(f)
   
def normalize_s(s, scale=STATE_NORM, single=False):
    if not single:
        s = scale[1] + s/scale[0]
    else:
        for i in range(len(scale)):
            s[i] = scale[i][1] + s[i]/scale[i][0]
            
    return (s)


def Test_Agent(actor, env, render=False):
    ''' Agent for policy validation after training '''            
    total_reward = 0; Act1 = []; Act2 = []
    s_t = env.reset()
    s_t = normalize_s(s_t)
    done =False
    while not done:
        if render:
            env.render()
        a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
        a_t[0][0] = a_t[0][0] * A_MAX[0][0]  + A_MAX[1][0]
        a_t[0][1] = a_t[0][1] * A_MAX[0][1]  + A_MAX[1][1]
        #Correct scaling!
        s_t, r_t, done, info = env.step(a_t[0])
        s_t = normalize_s(s_t)
        total_reward += r_t
        Act1.append(a_t[0][0])
        Act2.append(a_t[0][1])
    
    print("Episode:" + str(done_ctr) +": Reward " + str(total_reward)) 
    
    return (total_reward, Act1, Act2)


  
if __name__ == "__main__":
    
    #np.random.seed(17)

    reward = 0
    done = False
    step = 0
    epsilon = []

    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)
    
    f, ax = plt.subplots(4,1)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    noise1 = OrnsteinUhlenbeckActionNoise(1, mu = 0, theta = 0.15, sigma = 0.75)
    noise2 = OrnsteinUhlenbeckActionNoise(1, mu = 0, theta = 0.15, sigma = 0.75)
    
    # Get environment
    env = Plant()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    A_MAX = env.action_range
    
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    
    if LoadWeithsAndTrain == True: 
        try:
        		 ram = TempConfig.load_pickle("Memory_Rnd")
        		 print('Random memory loaded')
        except:
        		 print('Error while loading random memory')
                 
#        print("Now we load the weight")
#        try:
#            actor.model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_actor_model.h5")
#            critic.model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_critic_model.h5")
#            actor.target_model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_actor_model.h5")
#            critic.target_model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_critic_model.h5")
#            print("Weight load successfully")
#        except:
#            print("Cannot find the weight")

    print("Experiment Start.")
    R = []; step_ctr = 0
    try:  
        if LoadWeithsAndTest == False:
            #Train policiy
            for i in range(episode_count):
                print("Episode " + str(i) + "/"+ str(episode_count) +" Replay Buffer " + str(buff.count()) + "/"+ str(BUFFER_SIZE) )
        
                ob = env.reset()
        
                s_t = normalize_s(ob)
                
                Act1=[];   Act2=[]; L=[]
                total_reward = 0
                for j in range(max_steps):                    
                    loss = 0 
                    step_ctr +=1
                    epsilon.append(MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * step_ctr))
                    a_t = np.zeros([1,action_dim])
                    a_t_ = np.zeros([1,action_dim])
                    noise_t = np.zeros([1,action_dim])
                    
                    a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
                    
                    if epsilon[-1] > MIN_EPSILON + 0.02:
                        noise_t[0][0] =  max(epsilon[-1], 0) * noise1.sample() #OU.function(a_t_original[0][0],  0.5 , 0.80, 0.5) #x,mu,teta,sigma
                        noise_t[0][1] =  max(epsilon[-1], 0) * noise2.sample() #OU.function(a_t_original[0][1],  0.5 , 0.80, 0.5)
                    else:
                        if random.random() < MIN_EPSILON:
                            noise_t[0][0] =  0.5 * noise1.sample() 
                            noise_t[0][1] =  0.5 * noise2.sample()
                        else:                                                                                                                                                                                                                                               
                            noise_t[0][0] = 0.0 * noise1.sample()
                            noise_t[0][1] = 0.0 * noise1.sample()
                        
                    # Scale carefully actions suitable to environment
                    a_t_[0][0] = np.clip( a_t_original[0][0] + noise_t[0][0], -1, 1 ) 
                    a_t_[0][1] = np.clip( a_t_original[0][1] + noise_t[0][1], -1, 1 )
                    a_t[0][0] = a_t_[0][0] * A_MAX[0][0]  + A_MAX[1][0]
                    a_t[0][1] = a_t_[0][1] * A_MAX[0][1]  + A_MAX[1][1]
        
                    s_t1, r_t, done, info = env.step(a_t[0])                                                                                                                                                                    
                    r_t = np.clip(r_t, REW_MIN, 1) #-9, 1
                    s_t1 = normalize_s(s_t1)
                    
                    buff.add(s_t, a_t_[0], r_t, s_t1, done)      #Add replay buffer
                    
                    #Do the batch update
                    batch = buff.getBatch(BATCH_SIZE)
                    states = np.asarray([e[0] for e in batch])
                    actions = np.asarray([e[1] for e in batch])
                    rewards = np.asarray([e[2] for e in batch])
                    new_states = np.asarray([e[3] for e in batch])
                    dones = np.asarray([e[4] for e in batch])
                    y_t = np.asarray([e[1] for e in batch])
        
                    target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
                   
                    for k in range(len(batch)):
                        if dones[k]:
                            y_t[k] = rewards[k]
                        else:
                            y_t[k] = rewards[k] + GAMMA*target_q_values[k]
               
                    
                    loss += critic.model.train_on_batch([states,actions], y_t)
                    a_for_grad = actor.model.predict(states)
                    grads = critic.gradients(states, a_for_grad)
                    actor.train(states, grads)
                    actor.target_train()
                    critic.target_train()
        
                    total_reward += r_t
                    s_t = s_t1
                
                    #print("Episode", i, "Step", step, "Action", a_t[0])#, "Reward", r_t, "Loss", loss)
                    
                    Act1.append(a_t[0][0])
                    Act2.append(a_t[0][1])
                    L.append(loss)
                    
                    step += 1
                    if done:
                        plot_logs(f, ax, R, Act1, Act2, L, epsilon)
                        env.render()   
                        if LoadWeithsAndTest == False:
                            TempConfig.dump_pickle(buff, 'Memory_Rnd')
                        
                        if epsilon[-1]<0.2:
                            if total_reward > max(R) :
                                print('Best models saved')
                                TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_actor_model_best.h5", actor)
                                TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDQG_critic_model_best.h5", critic)
                                
                            if i % 50 == 0:
                                print('Checkpoint model saved at episode:', i)
                                TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_actor_model_cp_"+str(i)+".h5", actor)
                                TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_critic_model_cp_"+str(i)+".h5", critic)                            
                        break
        
                R.append(total_reward)
                
                print("TOTAL REWARD @ " + str(i) +"-th Episode  :" + str(total_reward))
                print("Total Step: " + str(step))
                print("")
            
            TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_actor_model_final.h5", actor)
            TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_critic_model_final.h5", critic)
            
            env.close()  

            print("Finish.")
            
        else:
            # Test policiy
            import glob
            scores = {}
            
            allWeights_actor = sorted(glob.glob(TempConfig.ModelsPath + "/Plant_DDPG_actor_model_cp*.h5"))
            allWeights_critic = sorted(glob.glob(TempConfig.ModelsPath + "/Plant_DDPG_critic_model_cp*.h5"))
            
            print('Load agent and play')
            
            for i in range(len(allWeights_actor)):
                actor.model.load_weights(allWeights_actor[i])
                critic.model.load_weights(allWeights_critic[i])
                #actor.model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_actor_model.h5")
                #critic.model.load_weights(TempConfig.ModelsPath+"Plant_DDPG_critic_model.h5")
                
                done_ctr = 0; 
                L=[]
                while done_ctr < 1 :
                    total_reward, Act1, Act2 = Test_Agent(actor, env)
                    R.append(total_reward) 
                    scores[allWeights_actor[i]]=total_reward
                    if total_reward >= max(scores.values()):
                        best_act = allWeights_actor[i]
                        best_crit = allWeights_critic[i]
    
                    #plot_logs(f, ax, R, Act1, Act2, L, epsilon)   
                    done_ctr += 1
                    
            print('**** Best score:', max(scores.values()), ' at cp:', best_act, '****')
            
            #render very best
            actor.model.load_weights(best_act)
            critic.model.load_weights(best_crit)
            total_reward, Act1, Act2 = Test_Agent(actor, env, render=True)
            plot_logs(f, ax, R, Act1, Act2, L, epsilon)   
                
            env.close()            
            
            
    except KeyboardInterrupt:
        print('User interrupt')
        env.close()
        plot_logs(f, ax, R, Act1, Act2, L, epsilon)  
        if LoadWeithsAndTest == False:
             print('Save model: Y or N?')
             save = input()
             if save.lower() == 'y':
                 TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_actor_model.h5", actor)
                 TempConfig.save_DDQL(TempConfig.ModelsPath, "Plant_DDPG_critic_model.h5", critic)
             else:
                print('Model discarded')
