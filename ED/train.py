import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import numpy as np
import gymnasium as gym
import random
import pickle
from symphony import Symphony
import math
import os, re

np.set_printoptions(threshold=10000, linewidth=200)

def seed_reset():
    r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
    torch.manual_seed(r1)
    np.random.seed(r2)
    random.seed(r3)
    return r1, r2, r3

#==============================================================================================
#==============================================================================================
#=========================================LOGGING=============================================
#==============================================================================================
#==============================================================================================

# to continue writing to the same history file and derive its name. This function created with the help of ChatGPT


def extract_r1_r2_r3():
    pattern = r'history_(\d+)_(\d+)_(\d+)\.log'

    # Iterate through the files in the given directory
    for filename in os.listdir():
        # Match the filename with the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the numbers r1, r2, and r3 from the filename
            return map(int, match.groups())
    return None


#write or append to the history log file
class LogFile(object):
    def __init__(self, log_name_main, log_name_opt):
        self.log_name_main = log_name_main
        self.log_name_opt = log_name_opt
    def write(self, text):
        with open(self.log_name_main, 'a+') as file:
            file.write(text)
    def write_opt(self, text):
        with open(self.log_name_opt, 'a+') as file:
            file.write(text)
    def clean(self):
        with open(self.log_name_main, 'w') as file:
            file.write("")
        with open(self.log_name_opt, 'w') as file:
            file.write("")


numbers = extract_r1_r2_r3()
if numbers != None:
    # derive random numbers from history file
    r1, r2, r3 = numbers
else:
    # generate new random seeds
    r1, r2, r3 = seed_reset()



print(r1, ", ", r2, ", ", r3)

log_name_main = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_name_opt = "episodes_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name_main, log_name_opt)


#==============================================================================================
#==============================================================================================
#===================================SCRIPT FOR TRAINING========================================
#==============================================================================================
#==============================================================================================



#global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
G = 2
learning_rate = 5e-5
explore_time, times = 7680, 50
capacity = explore_time * times
h_dim = capacity//1000
limit_step = 1000 #max steps per episode
limit_eval = 1000 #max steps per evaluation
num_episodes = 1000000
start_episode = 1 #number for the identification of the current episode
episode_rewards_all, episode_steps_all, test_rewards, Q_learning, total_steps = [], [], [], False, 0

# environment type.
option = 3
pre_valid = True
if option == 0: env_name = '"BipedalWalker-v3'
elif option == 1: env_name = 'HalfCheetah-v4'
elif option == 2: env_name = 'Walker2d-v4'
elif option == 3: env_name = 'Humanoid-v4'
elif option == 4: env_name = 'Ant-v4'
elif option == 5: env_name = 'Swimmer-v4'
elif option == 6: env_name = 'Hopper-v4'
elif option == 7: env_name = 'Pusher-v4'

env = gym.make(env_name)
env_test = gym.make(env_name)
env_valid = gym.make(env_name, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]
#max_action = torch.FloatTensor(env.action_space.high) if env.action_space.is_bounded() else torch.ones(action_dim)
max_action = torch.ones(action_dim)

print("action_dim: ", action_dim, "state_dim: ", state_dim, "max_action:", max_action)

algo = Symphony(capacity, state_dim, action_dim, h_dim, device, max_action, learning_rate)




#==============================================================================================
#==============================================================================================
#==========================================TESTING=============================================
#==============================================================================================
#==============================================================================================

#testing model
def testing(env, limit_step, test_episodes, current_step=0, save_log=False):
    if test_episodes<1: return
    print("Validation... ", test_episodes, " epsodes")
    episode_return = []

    for test_episode in range(test_episodes):

        state = env.reset()[0]
        rewards = []

        for steps in range(1,limit_step+1):
            action = algo.select_action(state, noise=False)
            next_state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            state = next_state
            if done or truncated: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode+1}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")

    if save_log: log_file.write(str(current_step) + " : " + str(round(validate_return.item(), 2)) +  "\n")


#==============================================================================================
#==============================================================================================
#=====================LOADING EXISTING MODELS, BUFFER and PARAMETERS===========================
#==============================================================================================
#==============================================================================================



def save_data_models():
    print("saving data...")
    torch.save(algo.nets.online.state_dict(), 'nets_model.pt')
    torch.save(algo.nets.target.state_dict(), 'nets_target_model.pt')
    torch.save(algo.nets_optimizer.state_dict(), 'nets_optimizer.pt')
    with open('data', 'wb') as file:
        pickle.dump({'buffer': algo.replay_buffer, 'q_next_ema': algo.nets.q_next_ema, 'episode_rewards_all':episode_rewards_all, 'episode_steps_all':episode_steps_all, 'total_steps': total_steps}, file)
    print("...saved")


try:
    print("loading buffer...")
    with open('data', 'rb') as file:
        dict = pickle.load(file)
        algo.replay_buffer = dict['buffer']
        algo.nets.q_next_ema = dict['q_next_ema']
        episode_rewards_all = dict['episode_rewards_all']
        episode_steps_all = dict['episode_steps_all']
        total_steps = dict['total_steps']
        if len(algo.replay_buffer)>=explore_time and not Q_learning: Q_learning = True
    print('buffer loaded, buffer length', len(algo.replay_buffer))

    start_episode = len(episode_steps_all)+1

except:
    print("problem during loading buffer")


try:
    print("loading models...")
    algo.nets.online.load_state_dict(torch.load('nets_model.pt', weights_only=True))
    algo.nets.target.load_state_dict(torch.load('nets_target_model.pt', weights_only=True))
    print('models loaded')
    if pre_valid: testing(env_valid, limit_eval, 100)
except:
    print("problem during loading models")


try:
    algo.nets_optimizer.load_state_dict(torch.load('nets_optimizer.pt', weights_only=True))
    print("optimizer loaded...")
except:
    print("problem during loading optimizer")
#==============================================================================================
#==============================================================================================
#========================================EXPLORATION===========================================
#==============================================================================================
#==============================================================================================



if not Q_learning:
    log_file.clean()
    
    while not Q_learning:
        rewards = []
        state = env_test.reset()[0]

        for steps in range(1, limit_step+1):

            seed_reset()
            action = algo.select_action(state)
            next_state, reward, done, truncated, info = env_test.step(action)
            rewards.append(reward)
            if algo.replay_buffer.length>=explore_time and not Q_learning: Q_learning = True; break
            algo.replay_buffer.add(state, action, reward, next_state, done)
            if done: break
            state = next_state

        Return = np.sum(rewards)
        print(f" Rtrn = {Return:.2f}")

    algo.replay_buffer.norm_fill(times)

    

#==============================================================================================
#==============================================================================================
#=========================================TRAINING=============================================
#==============================================================================================
#==============================================================================================


for i in range(start_episode, num_episodes):

    rewards = []
    state = env.reset()[0]
   

    for steps in range(1, limit_step+1):
        
        
        seed_reset()
        total_steps += 1
        
        # save models, data
        if (total_steps>=2500 and total_steps%2500==0):
            testing(env_test, limit_step=limit_eval, test_episodes=25, current_step=total_steps, save_log=True)
            if total_steps%5000==0: save_data_models()

   
        action = algo.select_action(state, action=(steps<limit_step-50))
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        algo.replay_buffer.add(state, action, reward, next_state, done)
        for _ in range(G): scale, beta = algo.train()
        if done: break
        state = next_state
        

    
    episode_rewards_all.append(np.sum(rewards))
    episode_steps_all.append(steps)
    
    print(f"Ep {i}: Rtrn = {episode_rewards_all[-1]:.2f} | ep steps = {steps} | total_steps = {total_steps}  | scale = {((scale)).mean().item():.4f} | beta = {((beta)).mean().item():.4f} ")
    log_file.write_opt(str(i) + " : " + str(round(episode_rewards_all[-1], 2)) + " : step : " + str(total_steps) + " : S : " + str(round(((scale)).mean().item(), 4)) + " : B : " + str(round(((beta)).mean().item(), 4)) + "\n")