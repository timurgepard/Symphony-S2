import logging
logging.getLogger().setLevel(logging.CRITICAL)
import torch
import numpy as np
import gymnasium as gym
import random
import pickle
import time
from symphony import Symphony, log_file


def seed_reset():
    r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
    torch.manual_seed(r1)
    np.random.seed(r2)
    random.seed(r3)


#==============================================================================================
#==============================================================================================
#===================================SCRIPT FOR TRAINING========================================
#==============================================================================================
#==============================================================================================



#global parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 3e-4
update_to_data = 3
explore_time = 5120
limit_step = 1000 #max steps per episode
limit_eval = 1000 #max steps per evaluation
num_episodes = 1000000
start_episode = 1 #number for the identification of the current episode
episode_rewards_all, episode_steps_all, test_rewards, Q_learning, total_steps = [], [], [], False, 0

# environment type.
option = 3
pre_valid = False


if option == 1:
    env = gym.make('HalfCheetah-v4', render_mode="human")
    env_test = gym.make('HalfCheetah-v4')

elif option == 2:
    env = gym.make('Walker2d-v4')
    env_test = gym.make('Walker2d-v4')

elif option == 3:
    env = gym.make('Humanoid-v4')
    env_test = gym.make('Humanoid-v4')

elif option == 4:
    env = gym.make('Ant-v4')
    env_test = gym.make('Ant-v4')

elif option == 5:
    burst = True
    env = gym.make('Swimmer-v4')
    env_test = gym.make('Swimmer-v4')

elif option == 6:
    env = gym.make('Hopper-v4')
    env_test = gym.make('Hopper-v4')


state_dim = env.observation_space.shape[0]
action_dim= env.action_space.shape[0]
#max_action = torch.FloatTensor(env.action_space.high) if env.action_space.is_bounded() else 1.0
max_action = torch.ones(action_dim)

print("action_dim: ", action_dim, "state_dim: ", state_dim, "max_action:", max_action)

algo = Symphony(state_dim, action_dim, device, max_action, learning_rate, update_to_data)




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
            seed_reset()
            action = algo.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            rewards.append(reward)
            state = next_state
            if done or truncated: break

        episode_return.append(np.sum(rewards))

        validate_return = np.mean(episode_return[-100:])
        print(f"trial {test_episode+1}:, Rtrn = {episode_return[test_episode]:.2f}, Average 100 = {validate_return:.2f}, steps: {steps}")

    if save_log: log_file.write(str(current_step) + " : " + str(round(validate_return.item(), 2)) + "\n")


#==============================================================================================
#==============================================================================================
#=====================LOADING EXISTING MODELS, BUFFER and PARAMETERS===========================
#==============================================================================================
#==============================================================================================





try:
    print("loading buffer...")
    with open('data', 'rb') as file:
        dict = pickle.load(file)
        algo.replay_buffer = dict['buffer']
        algo.q_next_ema = dict['q_next_ema']
        episode_rewards_all = dict['episode_rewards_all']
        episode_steps_all = dict['episode_steps_all']
        total_steps = dict['total_steps']
        if len(algo.replay_buffer)>=explore_time and not Q_learning: Q_learning = True
    print('buffer loaded, buffer length', len(algo.replay_buffer))

    start_episode = len(episode_steps_all)

except:
    print("problem during loading buffer")


try:
    print("loading models...")
    algo.nets.load_state_dict(torch.load('nets_model.pt', weights_only=True))
    algo.nets_target.load_state_dict(torch.load('nets_target_model.pt', weights_only=True))
    print('models loaded')
    if pre_valid: testing(env_test, limit_eval, 100)
except:
    print("problem during loading models")


def save_data_models():
    print("saving data...")
    torch.save(algo.nets.state_dict(), 'nets_model.pt')
    torch.save(algo.nets_target.state_dict(), 'nets_target_model.pt')
    with open('data', 'wb') as file:
        pickle.dump({'buffer': algo.replay_buffer, 'q_next_ema': algo.q_next_ema, 'episode_rewards_all':episode_rewards_all, 'episode_steps_all':episode_steps_all, 'total_steps': total_steps}, file)
    print("...saved")

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
            action = algo.select_action(state, explore=True)
            next_state, reward, done, truncated, info = env_test.step(action)
            rewards.append(reward)
            if algo.replay_buffer.length>=explore_time and not Q_learning: Q_learning = True; break
            algo.replay_buffer.add(state, action, reward, next_state, done)
            if done: break
            state = next_state

        Return = np.sum(rewards)
        print(f" Rtrn = {Return:.2f}")


#==============================================================================================
#==============================================================================================
#=========================================TRAINING=============================================
#==============================================================================================
#==============================================================================================



print("started training")


for i in range(start_episode, num_episodes):

    rewards = []
    state = env.reset()[0]
    #--------------------2. CPU/GPU cooling ------------------
    time.sleep(0.3)

    for steps in range(1, limit_step+1):

        algo.train()
        seed_reset()

        total_steps += 1

        # save models, data
        if (total_steps>=1250 and total_steps%1250==0):
            testing(env_test, limit_step=limit_eval, test_episodes=50, current_step=total_steps, save_log=True)
            if total_steps%10000==0: save_data_models()


        action = algo.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        algo.replay_buffer.add(state, action, reward, next_state, done)
        if done: break
        state = next_state

    episode_rewards_all.append(np.sum(rewards))
    episode_steps_all.append(steps)


    print(f"Ep {i}: Rtrn = {episode_rewards_all[-1]:.2f} | ep steps = {steps} | total_steps = {total_steps}")

    log_file.write_opt(str(i) + " : " + str(round(episode_rewards_all[-1], 2)) + " : step : " + str(total_steps) + "\n")
