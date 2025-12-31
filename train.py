from symphony import Symphony
import gymnasium as gym

import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import numpy as np
import random
import pickle
import time
import os, re

#############################################
# -----------Helper Functions---------------#
#############################################



# random seeds for reproducing the experiment
def seed_reset():
    r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
    torch.manual_seed(r1)
    np.random.seed(r2)
    random.seed(r3)
    return r1, r2, r3


def extract_r1_r2_r3():
    pattern = r'history_(\d+)_(\d+)_(\d+)\.csv'

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
            file.write("step,return\n")
        with open(self.log_name_opt, 'w') as file:
            file.write("ep,return,steps,scale\n")


numbers = extract_r1_r2_r3()

if numbers != None:
    # derive random numbers from history file
    r1, r2, r3 = numbers
else:
    # generate new random seeds
    r1, r2, r3 = seed_reset()



print(r1, ", ", r2, ", ", r3)

log_name_main = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".csv"
log_name_opt = "episodes_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".csv"
log_file = LogFile(log_name_main, log_name_opt)


def save(algo, total_rewards, total_steps):

    torch.save(algo.nets.online.state_dict(), 'nets_online_model.pt')
    torch.save(algo.nets.target.state_dict(), 'nets_target_model.pt')
    torch.save(algo.nets_optimizer.state_dict(), 'nets_optimizer.pt')
    print("saving... the buffer length = ", algo.replay_buffer.length, end="")
    with open('data', 'wb') as file:
        pickle.dump({'buffer': algo.replay_buffer, 'q_next_ema': algo.nets.q_next_ema, 'total_rewards': total_rewards, 'total_steps': total_steps}, file)
    print(" > done")


def load(algo, Q_learning):

    total_rewards, total_steps = [], 0

    try:
        print("loading models...")
        algo.nets.online.load_state_dict(torch.load('nets_online_model.pt', weights_only=True))
        algo.nets.target.load_state_dict(torch.load('nets_target_model.pt', weights_only=True))
        algo.nets_optimizer.load_state_dict(torch.load('nets_optimizer.pt', weights_only=True))
        print('models loaded')
        sim_loop(env_valid, 100, True, False, algo, [], total_steps=0)
    except:
        print("problem during loading models")


    try:
        print("loading buffer...")
        with open('data', 'rb') as file:
            dict = pickle.load(file)
            algo.replay_buffer = dict['buffer']
            algo.nets.q_next_ema = dict['q_next_ema']
            total_rewards = dict['total_rewards']
            total_steps = dict['total_steps']
            if algo.replay_buffer.length>=explore_time and not Q_learning: Q_learning = True
        
        print('buffer loaded, Q_ema', round(algo.nets.q_next_ema.item(), 2), ', average_reward = ', round(np.mean(total_rewards[-300:]), 2))
        
    except:
        print("problem during loading buffer")

    return Q_learning, total_rewards, total_steps

#############################################
# ---------------Parametres-----------------#
#############################################

#global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

print(device)
G = 3
learning_rate = 5e-5
explore_time, times = 20480, 25
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


# Loop for episodes:[ State -> Loop for one episode: [ Action, Next State, Reward, Done, State = Next State ] ]
def sim_loop(env, episodes, testing, Q_learning, algo, total_rewards, total_steps):


    start_episode = len(total_rewards) + 1


    for episode in range(start_episode, episodes+1):
            
        Return = 0.0     
        state = env.reset()[0]
        
        for steps in range(1,limit_step+1):

            seed_reset()
            total_steps += 1

            # Activate training if explore time is reached and if it is not testing mode:
            if testing:
                Q_learning = False
            else:
                if algo.replay_buffer.length>=explore_time and not Q_learning:
                    Q_learning = True
                    algo.replay_buffer.norm_fill(times)
                    print("started training")

            # if total steps is divisible to 2500 save models, stop training and do testing, return to training:
            if Q_learning and total_steps>=2500 and total_steps%2500==0:
                save(algo, total_rewards, total_steps)
                
                print("start testing")
                test_return = sim_loop(env_test, 25, True, Q_learning, algo, [], total_steps=0)
                log_file.write(str(total_steps) + "," + str(round(test_return, 2)) + "\n")
                print("end of testing")


            # if steps is close to episode limit (e.g. 950) we shut down actions and leave noise to get Terminal Transition:
            active = steps<(limit_step-50) if Q_learning else True
            action = algo.select_action(state,  action=active, noise=not testing)
            next_state, reward, done, truncated, info = env.step(action)
            if not testing: algo.replay_buffer.add(state, action, reward, next_state, done)
            Return += reward
            
            # actual training
            if Q_learning: [scale := algo.train() for _ in range(G)]
            if done or truncated: break
            state = next_state



        total_rewards.append(Return)
        average_reward = np.mean(total_rewards[-300:])


        print(f"Ep {episode}: Rtrn = {Return:.2f}, Avg = {average_reward:.2f}| ep steps = {steps} | total_steps = {total_steps}") 
        if not testing and Q_learning: log_file.write_opt(str(episode) + "," + str(round(Return, 2)) + "," + str(total_steps) + "," + str(round(scale.mean().item(), 4)) + "\n")
        

    return np.mean(total_rewards).item()




# Loading existing models
Q_learning, total_rewards, total_steps = load(algo, Q_learning)
if not Q_learning: log_file.clean()

# Training
sim_loop(env, num_episodes, False, Q_learning, algo, total_rewards, total_steps)