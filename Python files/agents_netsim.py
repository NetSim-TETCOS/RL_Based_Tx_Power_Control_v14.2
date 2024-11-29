import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3
import torch
import matplotlib.pyplot as plt
import random
# Assuming your custom environment class is defined as `g3_ue6` in a file named `env_21Jun2024.py`
from env_general import generalized
import tempfile
import csv
import os

# in Case your pc lags
# import psutil
# import os
# p = psutil.Process(os.getpid())
# p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

# Create a function to make the environment


seed = 31
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Get the path to the temporary directory
# temp_dir = tempfile.gettempdir()
# print("Temporary directory:", temp_dir)
# file_name = "DeviceCount.csv"
# file_path = os.path.join(temp_dir, file_name)
# # Read the CSV file
# if os.path.exists(file_path):
#     with open(file_path, 'r', newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         called_once = False
#         for row in reader:
#             if(not called_once):
#                called_once = True
#             else:
#                 # Assuming the file has one row and two columns
#                 value1, value2 = row
#                 print("Value 1:", value1)
#                 print("Value 2:", value2)
# else:
#     print(f"The file {file_name} does not exist in the temporary directory.")


num_gNBs = (int)(input("Enter number of gNBS "))
num_UEs = (int)(input("Enter number of UES "))

port_num=(int)(input("Enter port "))
# Specify number of episodes we need to run for
episodes=1500
steps=500
# num_gNBs = (int)(value2)
# print("number of gNBs are", num_gNBs)
# num_UEs = (int)(value1)
# print("number of UEs are", num_UEs)
# Wrap the environment
# way 1
# env = DummyVecEnv([make_env])  
# way 2
# env = gym.make('general-v1',num_gNBs=num_gNBs,num_UEs=num_UEs,port_num=port_num)
env = DummyVecEnv([lambda: gym.make('general-v2', num_gNBs=num_gNBs, num_UEs=num_UEs, port_num=port_num,seed_val=seed,num_ep=episodes,num_step=steps)])
# env = VecNormalize(env, norm_obs=True, norm_reward=True)
total_rewards_per_episode=[]

# Custom linear/noop activation function
class Identity(torch.nn.Module):
    def forward(self, x):
        return x
policy_kwargs = dict(
    net_arch=[64, 64],  # Two hidden layers with 64 neurons each
    activation_fn=Identity  # Use preset/custom function
    # activation_fn=torch.nn.Tanh
    # activation_fn=torch.nn.LeakyReLU
    # default activation for A2C and PPO is Tanh
    # Linear advised for PPO,ReLU/Linear for DQN0.
)

# C.reate the agent DQN / A2C / PPO
# model = A2C(
#     "MlpPolicy", 
#     env, 
#     policy_kwargs = policy_kwargs,
#     n_steps=20,
#     learning_rate=0.002,
#     # ent_coef=0.002,
#     verbose=0 #to avoid detailed output set verbose=0 else verbose=1
# )
model = PPO(
    "MlpPolicy", 
    env, 
    policy_kwargs = policy_kwargs,
    n_steps=500, 
    verbose=0) #to avoid detailed output set verbose=0 else verbose=1
# model = DQN(
#     "MlpPolicy",
#     env, 
#     policy_kwargs = policy_kwargs,
#     batch_size=64,
#     target_update_interval=5000,
#     learning_starts=200, 
#     exploration_fraction=0.2,
#     # learning_rate=1e-3,
#     verbose=0) #to avoid detailed output set verbose=0 else verbose=1

# Training the model
print("model learning")
try:
    model.learn(total_timesteps=episodes*steps)
    # for i in range(episodes):
    #     print(f"Episode: {i}")
    #     # env.reset()
    #     model.learn(total_timesteps=steps)
    #     # env.unwrapped.resetcon()
    #     print("episode end")
    #     total_rewards_per_episode.append(sum(env.get_attr('total_rewards')[0])/(steps))
    #     print(total_rewards_per_episode[-1])
        # env.reset()
except KeyboardInterrupt:
        print("Loop terminated by user.") #If you want to stop training in between 
# model.learn(total_timesteps=episodes*steps)

total_rewards_per_episode=env.get_attr("total_rewards_per_ep")[0]
plt.plot(total_rewards_per_episode)  # EMA
plt.title("PPO Algorithm, 1500 episodes * 500 iterations; Activation: Identity")
plt.xlabel("Episodes")
plt.ylabel("Average Sum Throughput(Mbps)")
plt.savefig("Average_rewards_ppo_1500x500_identity_7x20.png")
plt.show()

# Saving the model
model_path = "a2c_g3_ue6_model_general.zip" # will be saved in the directory selected at runtime
# To give custom path
# model_path = "C:/Users/rpamn/Desktop/Ronak/ql new/openai gym/21Jun2024/ppo_g3_ue6_model_2kx1.5k_env3.zip"
# model.save(model_path)
# del model
# print(f"Model saved to {model_path}.")

# model=A2C.load(model_path)

# # Evaluate the trained agent
# obs = env.reset()
# for _ in range(10):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     obs_tensor = torch.tensor(obs, dtype=torch.float64)
#     # In case you use DQN
#     # q_values = model.policy.q_net(obs_tensor.reshape(1, -1)).detach().numpy()
#     # print("Q-values:", q_values)
#     print("Chosen action:", action)
#     print(rewards)
#     print(obs)



