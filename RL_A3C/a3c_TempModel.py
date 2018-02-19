"""
Reinforcement Learning.
A3C with continuous action space.
"""
#tensorboard --logdir="./log" --port6006
import math
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from Env_Plant import Plant
import os
import shutil
import matplotlib.pyplot as plt

LoadWeithsAndTest = False

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 160000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.5	#0.01
LR_A = 0.00001    	# learning rate for actor
LR_C = 0.0001    	# learning rate for critic
GLOBAL_EP = 0
STATE_NORM = (10, 1) #Normalize observations: (1/gain, offset)

env = Plant()
MAX_EPSILON = ENTROPY_BETA
MIN_EPSILON = 0.001

EXPLORATION_STOP = int(400 * env.Plant_MaxSteps)    # at this step epsilon will be MIN_EPSILON
LAMBDA = - math.log(0.01) / EXPLORATION_STOP        # speed of decay fn of episodes of learning agent

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
ACTION_BOUND = env.ACTION_BOUND 
ACTION_GAP = env.ACTION_GAP

# Global variables for plotting 
GLOBAL_RUNNING_R = []; GLOBAL_ACTOR_LOSS = []; GLOBAL_CRITIC_LOSS = [] 
GLOBAL_ACT1 = []; GLOBAL_ACT2 = []; GLOBAL_ACT11 = []; GLOBAL_ACT22 = []

def plot_logs(f, ax1, reward, act1=[], act2=[], act11=[], act22=[], loss_A=[], loss_C=[]):
    
    f.set_figwidth(10)
    f.set_figheight(10)

    ax1[0].set_title('Reward')
    #ax1.set_xlim([0,50])
    ax1[0].plot(reward, label= 'Reward')
    if len(reward) > 10:
        mav = np.convolve(reward, np.ones((10,))/10, mode='valid')
        ax1[0].plot(mav, label= 'MovingAv10')
    ax1[0].legend(loc='upper left')
   
    ax1[1].set_title('Worker 0')
    ax1[1].plot(act1, label= 'action1')  
    ax1[1].plot(act2, label= 'action2')
    ax1[1].legend(loc='upper right')
    
    ax1[2].set_title('Worker 1')
    ax1[2].plot(act11, label= 'action1')  
    ax1[2].plot(act22, label= 'action2')
    ax1[2].legend(loc='upper right')
    
    ax1[3].plot(loss_A, label= 'Epsilon') 
    ax1[3].plot(loss_C, label= 'Critic loss')  
    ax1[3].legend(loc='upper right')
#
#    ax1[2].plot(eps, label= 'Epsilon')    
#    ax1[2].legend(loc='upper right')
    
    f.savefig('RL_A3C_Plant.png')
    ax1[0].cla(); ax1[1].cla(); ax1[2].cla(); ax1[3].cla()
    #ax1.cla()
    plt.close(f)

def normalize_s(s, scale=STATE_NORM, single=False):
    if not single:
        s = scale[1] + s/scale[0]
    else:
        for i in range(len(scale)):
            s[i] = scale[i][1] + s[i]/scale[i][0]
            
    return (s)    
    
class ACNet(object):
    ''' Actor critic network - build global and local graphs'''
    def __init__(self, scope, globalAC=None):
        self.step = 1
        self.epsilon = [MAX_EPSILON]
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * ACTION_GAP + ACTION_BOUND, sigma * ACTION_GAP * 0.05 + 1e-4

                    
                normal_dist =  tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = 0.5*(log_prob) * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration

                    self.exp_v = self.epsilon[-1] * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    #self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), -200, 200)
                    self.A = normal_dist.sample(1)[0]
                    
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    
            with tf.name_scope('sync_'+scope):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        w_init1 = tf.random_normal_initializer(0., .2)
        w_init2 = tf.random_normal_initializer(0., .05)
        w_init3 = tf.random_normal_initializer(0., .15)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 500, tf.nn.tanh, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(l_a, 500, tf.nn.tanh, kernel_initializer=w_init1, name='la_')
            l_a1 = tf.layers.dense(l_a, 500, tf.nn.tanh, kernel_initializer=w_init2, name='la__')
            l_a = tf.layers.dense(l_a1, 500, tf.nn.tanh, kernel_initializer=w_init2, name='la___')
            #l_a = tf.layers.dense(l_a, 200, tf.nn.tanh, kernel_initializer=w_init2, name='la____')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init1, name='sigma') #tf.nn.softplus

        with tf.variable_scope('critic'):
            #l_c = tf.layers.dense(self.s, 300, tf.nn.tanh, kernel_initializer=w_init3, name='lc')
            #l_c = tf.layers.dense(l_c, 300, tf.nn.tanh, kernel_initializer=w_init1, name='lc_')
            v = tf.layers.dense(l_a1, 1, kernel_initializer=w_init3, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        self.step += 1
        self.epsilon.append(MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.step))
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]


class Worker(object):
    ''' Asynchronous worker '''
    def __init__(self, name, globalAC, mutex):
        self.env = Plant()
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.mutex = mutex

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        global GLOBAL_ACT1, GLOBAL_ACT2, GLOBAL_ACT11, GLOBAL_ACT22
        
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            s = normalize_s(s)
            ep_r = 0
            while True:
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                s_ = normalize_s(s_)
                
                if self.name == 'W_0':
                    GLOBAL_ACT1.append(a[0])
                    GLOBAL_ACT2.append(a[1])
                if self.name == 'W_1':
                    GLOBAL_ACT11.append(a[0])
                    GLOBAL_ACT22.append(a[1])
                
                r = np.clip(r, -20, 1)    
                ep_r += r
                
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+21)/20)    # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    self.mutex.acquire()  # Note: python + = is not an atomic operation
                    self.env.render()
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(self.name,"Ep:", GLOBAL_EP,"| Ep_r: %i" % GLOBAL_RUNNING_R[-1], 'step:', self.AC.step)
                    GLOBAL_EP += 1
                    
                    plot_logs(f, ax, GLOBAL_RUNNING_R, GLOBAL_ACT1, GLOBAL_ACT2, GLOBAL_ACT11, GLOBAL_ACT22, self.AC.epsilon, GLOBAL_CRITIC_LOSS)
                    if self.name == 'W_0':
                        GLOBAL_ACT1 = []; GLOBAL_ACT2 = []
                    if self.name == 'W_1':
                        GLOBAL_ACT11 = []; GLOBAL_ACT22 = []
                        
                    self.mutex.release()
                    break

class Worker_test(object):
    ''' Worker for testing after training '''
    def __init__(self, sess, name, globalAC, a_p, c_p):
        self.env = Plant()
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.AC.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.AC.a_params, a_p)]
        self.AC.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.AC.c_params, c_p)]
    
    def pull_global(self, sess):  
        sess.run([self.AC.pull_a_params_op, self.AC.pull_c_params_op])                            

def Test_Agent(sess, worker, render=False):
    ''' Function for testing trained policy '''            
    total_reward = 0; Act1 = []; Act2 = []
    s_t = worker.env.reset()
    s_t = normalize_s(s_t)
    done =False
    
    worker.pull_global(SESS)
    while not done:
        if render:
            worker.env.render()
        a = worker.AC.choose_action(s_t)
        s_t, r, done, info = worker.env.step(a)

        s_t = normalize_s(s_t)
        total_reward += r
        Act1.append(a[0])
        Act2.append(a[1])
    worker.env.render()
    
    print("Reward " + str(total_reward)) 
    
    return (total_reward, Act1, Act2)


if __name__ == "__main__":
    
    tf.reset_default_graph()
    tf.Graph().as_default()
    SESS = tf.Session()
    
    f, ax = plt.subplots(4,1)
    unique_mutex = threading.Lock()
    try:  
        if LoadWeithsAndTest == False:   
            # Train network 	 
            with tf.device("/cpu:0"):
                OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
                OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
                GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
                workers = []
                # Create worker
                for i in range(N_WORKERS):
                    i_name = 'W_%i' % i   # worker name
                    workers.append(Worker(i_name, GLOBAL_AC, unique_mutex))
        
            COORD = tf.train.Coordinator()
            SESS.run(tf.global_variables_initializer())
        
            if OUTPUT_GRAPH:
                if os.path.exists(LOG_DIR):
                    shutil.rmtree(LOG_DIR)
                writer = tf.summary.FileWriter(LOG_DIR, SESS.graph)
        
            worker_threads = []
            for worker in workers:
                job = lambda: worker.work()
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)
            COORD.join(worker_threads)
            
            plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
            plt.xlabel('step')
            plt.ylabel('Total moving reward')
            plt.show()
            
            writer.close()
            saver = tf.train.Saver(name="Saver")
            saver.save(SESS, "./KerasModels/tf_A3C_end")
            print('Session saved')
            
        else:
            # Test policiy
            print('Load agent and play')
            OPT_A = tf.train.RMSPropOptimizer(LR_A)
            OPT_C = tf.train.RMSPropOptimizer(LR_C)
            GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
            
            new_saver = tf.train.import_meta_graph("./KerasModels/tf_A3C_" +'.meta')
            new_saver.restore(SESS, tf.train.latest_checkpoint('./KerasModels/'))
            #graph = tf.get_default_graph()
            SESS.run(tf.global_variables_initializer())
            
            #op_to_restore = graph.get_name_scope()
            
            #GlobalNet_Params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GLOBAL_NET_SCOPE)
            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GLOBAL_NET_SCOPE + '/actor')
            c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GLOBAL_NET_SCOPE + '/critic')
            
            #GLOBAL_AC.a_params = [l_p.assign(g_p) for l_p, g_p in zip(GLOBAL_AC.a_params, a_params)]
            #GLOBAL_AC.c_params = [l_p.assign(g_p) for l_p, g_p in zip(GLOBAL_AC.c_params, c_params)]
            
            worker_t = Worker_test(SESS, 'worker_test', GLOBAL_AC, a_params, c_params)
            total_reward, Act1, Act2 = Test_Agent(SESS, worker_t)
            
            plot_logs(f, ax, GLOBAL_RUNNING_R, Act1, Act2)
                    
            
    except KeyboardInterrupt:
        print('User interrupt')
        env.close()
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('step')
        plt.ylabel('Total moving reward')
        plt.show()

        if LoadWeithsAndTest == False:
             print('Save model: Y or N?')
             save = input()
             if save.lower() == 'y':
                 saver = tf.train.Saver(name="Saver")
                 saver.save(SESS, "./KerasModels/tf_A3C_end")
                 print('Session saved')
             else:
                print('Model discarded')
        COORD.should_stop()
        COORD.join(worker_threads)
        
