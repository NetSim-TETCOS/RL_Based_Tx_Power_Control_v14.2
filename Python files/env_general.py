import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
import numpy as np
import torch
import random
import struct
import socket
import logging
import os 

num_UEs=11

#get the current working directory
# parent_dir = os.getcwd()

# # Logging
# directory_logs = "logs"
# path_logs = os.path.join(parent_dir, directory_logs)
# print(os.path.isdir(path_logs))
# if(not os.path.isdir(path_logs)):
#     os.mkdir(path_logs)

# logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
# all_values_logger = logging.getLogger('GNB_Powers_Logger')

# # Create a CSV file handler for logging
# log_filename = os.path.join(path_logs, "all_log.csv")
# all_values_csv_handler = logging.FileHandler(log_filename, mode='w')
# all_values_logger.addHandler(all_values_csv_handler)


# header_UE = ','.join([f'UE{i+1}_Throughput' for i in range(num_UEs)]) + '\n'
# header_SINR = ','.join([f'UE{i+1}_SINR' for i in range(num_UEs)]) + '\n'
# all_values_csv_handler.stream.write("Episode, Iteration,"+header_UE[:-1]+",Sum_Throughput,"+header_SINR[:-1]+"\n")

# def log_all_values(all_values):
#     all_values_logger.info(",".join(map(str, all_values)))


register(
    id="general-v2",
    entry_point="env_general:generalized" #filename:classname
)

class generalized(gym.Env):
    def __init__(self,num_gNBs=3,num_UEs=6,port_num=12345,seed_val=31,num_ep=1000,num_step=500):
        super(generalized, self).__init__()
        
        # State parameters
        self.it=0
        self.epi=0
        self.num_gNBs = num_gNBs
        self.num_UEs = num_UEs
        self.adjustments = [-3, -1, 0, 1, 3]
        # self.adjustments = [0, 0, 0, 0, 0]
        self.num_actions=len(self.adjustments) ** self.num_gNBs
        self.current_gNB_powers = [40 for _ in range(self.num_gNBs)]
        self.total_rewards=[]
        self.total_rewards_per_ep=[]
        # self.total_rewards=0
        self.num_ep=num_ep
        self.num_step=num_step
        # Define action space
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-500.00,
            high=500.00,
            shape=(self.num_UEs,),
            dtype=np.float64,
        )
        # Interfacing variables
        self.port = port_num
        self.server_address = (socket.gethostbyname(socket.gethostname()), self.port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.client_socket.connect(self.server_address)
        self.bytes_to_receive = (2*self.num_UEs+1)*8
        self.seed_val = seed_val
        self._seed(self.seed_val)

        # Initial state
        self.state = np.zeros(self.num_UEs)

        self.render_mode = None

    def _seed(self, seed=None):
        self.seed_val = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self._seed(seed if seed is not None else self.seed_val)
        if(self.it!=0):
            self.epi+=1
            self.total_rewards_per_ep.append((sum(self.total_rewards)/self.it))
            print("Episode : " + str(self.epi) + " Avg Reward : "+ str(self.total_rewards_per_ep[-1]))
            # self.dummy()
            self.client_socket.close()
        # Reset state
        self.current_gNB_powers = [40 for _ in range(self.num_gNBs)]
        self.total_rewards.clear()
        self.it=0
        # self.episode_Start()
        return self.state, {}
    
    def dummy(self):
        flag, next_sinr_values, throughputs_NETSIM, reward = self.NETSIM_interface()
        if flag == 0:
            print("flag 0 at itr",self.it)
            print("dummy caught")

    def episode_Start(self):
        while True:
            try:
                self.server_address = (socket.gethostbyname(socket.gethostname()), self.port)
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect(self.server_address)
                print("Connected to the server")
                break
            except:
                continue
        
        request_value = 0
        msgType = 0
        packed_value = struct.pack(">II",msgType, request_value)
        #send a request value to NetSim
        bytes_Sent = self.client_socket.send(packed_value)
        #recieve SINRs from NetSim
        data = self.client_socket.recv(self.bytes_to_receive)
        format_string = f'>{2*self.num_UEs+1}d'
        unpacked_data = struct.unpack(f'>{format_string[1:]}', data)

        self.state = unpacked_data[0:self.num_UEs]
        

    def step(self, action):
        if self.it==0:
            self.episode_Start()
        self.it+=1
        # print("iteration ", self.it)
        # Apply action
        delta_p = self.action_to_power_adjustments(action)
        self.current_gNB_powers = [p + dp for p, dp in zip(self.current_gNB_powers, delta_p)]
        self.current_gNB_powers = [max(27, min(46, p)) for p in self.current_gNB_powers]
        
        done = False
        # Additional info
        info = {}
        flag, next_sinr_values, throughputs_NETSIM, reward = self.NETSIM_interface() 
        # print(next_sinr_values)
        if flag == 0:
            # next_sinr_values = sinr_values
            # print("no flag received")
            print("flag 0 at itr",self.it)
            rew=self.total_rewards[-1]
            self.reset()
            # self.resetcon()
            # self.client_socket.close()
            return self.state, rew, done, False,info
        # Update state
        self.state = next_sinr_values
        self.total_rewards.append(reward)
        
        # log_all_values([throughputs_NETSIM[0],throughputs_NETSIM[1],throughputs_NETSIM[2],throughputs_NETSIM[3],throughputs_NETSIM[4],throughputs_NETSIM[5], reward, next_sinr_values[0],next_sinr_values[1],next_sinr_values[2],next_sinr_values[3],next_sinr_values[4],next_sinr_values[5], self.epi, self.it])
        # log_all_values([self.epi,self.it,*throughputs_NETSIM[0:self.num_UEs],reward,*next_sinr_values[0:self.num_UEs]])
        # print("iteration end")
        return self.state, reward, done, False,info

    def action_to_power_adjustments(self, action_index):
        delta_p = []
        for i in range(self.num_gNBs):
            delta_p.append(self.adjustments[action_index % len(self.adjustments)])
            action_index //= len(self.adjustments)
        return delta_p
    
    def get_spectral_eff(self,spectral_eff):
        for i in range(len(self.spectral_eff_list)-1,-1,-1):
            if(self.spectral_eff_list[i] <= spectral_eff):
                return self.spectral_eff_list[i]
        return 0

    def NETSIM_interface(self):
        
        dummy_array = np.zeros((self.num_UEs))

        msgType = 1
        # Construct the format string dynamically
        format_string = f'>{len(self.current_gNB_powers)}d'
        packed_data = struct.pack(f'>I{format_string[1:]}', msgType, *self.current_gNB_powers)

        try: 
            #sending a header value indicating that NetSim should recieve the gNB powers
            self.client_socket.send(packed_data)
        except:
            return 0,dummy_array,dummy_array,0
        
        try:
            #receive an acknowledgement message from NetSim
            data = self.client_socket.recv(4)
        except:
            return 0,dummy_array,dummy_array,0
            

        unpacked_data = struct.unpack('>I', data)

        msgType = 0
        request_value = 2
        packed_value = struct.pack(">II", msgType, request_value)

        try:
            #sending request to NetSim to send over new state and the reward
            self.client_socket.send(packed_value)
        except:
            return 0,dummy_array,dummy_array,0
        

        try:
            #receive the requested data

            data = self.client_socket.recv(self.bytes_to_receive)
            # print(data)
        except:
            return 0,dummy_array,dummy_array,0
        

        # print(len(data))
        format_string = f'>{2*self.num_UEs+1}d'
        unpacked_data = struct.unpack(f'>{format_string[1:]}', data)
        new_snirs = unpacked_data[0:self.num_UEs]
        new_throughputs = unpacked_data[self.num_UEs:2*self.num_UEs]

        return 1, new_snirs, new_throughputs, unpacked_data[2*self.num_UEs]

    #can be omitted
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Last total reward: {self.total_rewards[-1]}")
        elif mode == 'rgb_array':
            pass
        else:
            raise ValueError(f"Unknown render mode: {mode}")
        return None
    
    def resetcon(self):
        self.client_socket.close()

    def dB_to_linear(self, dBValue):
        power = (dBValue/10)
        linearValue = 10**(power)
        return linearValue
