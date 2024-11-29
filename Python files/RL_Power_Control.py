import numpy as np
import matplotlib.pyplot as plt
import socket
import struct
import logging
import pandas as pd
import time
import os
import tempfile
import csv

#get the current working directory
parent_dir = os.getcwd()

#makes a directory named "plots"
directory_plots = "plots"
path_plots = os.path.join(parent_dir, directory_plots)
print(os.path.isdir(path_plots))
if(not os.path.isdir(path_plots)):
  os.mkdir(path_plots)

#makes a directory named "logs"
directory_logs = "logs"
path_logs = os.path.join(parent_dir, directory_logs)
print(os.path.isdir(path_logs))
if(not os.path.isdir(path_logs)):
    os.mkdir(path_logs)

#registering the start time of the simulation
start_time = time.time()

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



port = 12349
num_gNBs = (int)(input("Enter number of gNBS"))
num_UEs = (int)(input("Enter number of UES"))

BW_MHz = 50
adjustments = [-3, -1, 0, 1, 3]  # Possible adjustments
# percentiles = [-11.550987805408719, -5.915791373545139, -0.0417750795523901]
percentiles = [-7,3]
num_UE_buckets = len(percentiles)+1
num_gNB_buckets = len(adjustments)
num_states = num_UE_buckets ** num_UEs  # 4096 states
num_actions = num_gNB_buckets ** num_gNBs  # 125 actions
q_table = np.zeros((num_states, num_actions))
gNB_powers = [40 for _ in range(num_gNBs)]
num_episodes = (int)(input("Enter the number of episodes\n"))
# num_episodes = 500
exploration = num_episodes/2
epsilon = 1 # not a constant, going to be decayed
EPSILON_DECAY = 0.25 ** (1 / exploration)
MIN_EPSILON = 0.25
gamma = 0.9
alpha = 0.3
bytes_to_receive = (2*num_UEs+1)*8

total_rewards_per_episode = []

dl_ul_ratio = 4/5
overhead_multiplier = 0.86
app_by_phy = 0.70
ue_divider = 0.5

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger('GNB_Powers_Logger')
sinr_logger = logging.getLogger('SINR_Logger')
all_values_logger = logging.getLogger('ALL_VALUE_Logger')

# Create a CSV file handler for logging gNB powers
log_filename = os.path.join(path_logs, "gnb_powers_log.csv")
gnb_powers_csv_handler = logging.FileHandler(log_filename, mode='w')
logger.addHandler(gnb_powers_csv_handler)

# Create a CSV file handler for logging SINR values
log_filename = os.path.join(path_logs, "sinr_values_log.csv")
sinr_csv_handler = logging.FileHandler(log_filename, mode='w')
sinr_logger.addHandler(sinr_csv_handler)

# Create a CSV file handler for logging throuhgputs and sinr values at each iteration 
log_filename = os.path.join(path_logs, "all_values_log.csv")
all_values_csv_handler = logging.FileHandler(log_filename, mode='w')
all_values_logger.addHandler(all_values_csv_handler)


# Write headers to the CSV files without newline characters
header_gNB = ','.join([f'gNB_Power_{i+1}' for i in range(num_gNBs)]) + '\n'
gnb_powers_csv_handler.stream.write(header_gNB)
header_UE = ','.join([f'UE{i+1}_Throughput' for i in range(num_UEs)]) + '\n'
sinr_csv_handler.stream.write(header_UE)
header_SINR = ','.join([f'UE{i+1}_SINR' for i in range(num_UEs)]) + '\n'
all_values_csv_handler.stream.write(header_UE[:-1]+",Sum_Throughput,"+header_SINR[:-1]+",Episode, Iteration\n")

#CQI table 
spectral_eff_list = [0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063]


#functions to log values in the respective files
def log_gnb_powers(gNB_powers):
    logger.info(",".join(map(str, gNB_powers)))

def log_sinr_values(sinr_values):
    sinr_logger.info(",".join(map(str, sinr_values)))

def log_all_values(all_values):
    all_values_logger.info(",".join(map(str, all_values)))


# Function to map an action index to power adjustments
def action_to_power_adjustments(action_index):
    
    delta_p = []
    for i in range(num_gNBs):
        delta_p.append(adjustments[action_index % num_gNB_buckets])
        action_index //= num_gNB_buckets
    return delta_p


#discretize the states based on the 25th, 50th, 75th percentiles of the sinr data
def get_state(sinr_values):
    state = 0
    for sinr in sinr_values:
        if sinr < percentiles[0]:
            state = state * num_UE_buckets + 0
        elif sinr < percentiles[1]:
            state = state * num_UE_buckets + 1
        else :
            state = state * num_UE_buckets + 2
    return state

def dB_to_linear(dBValue):
    power = (dBValue/10)
    linearValue = 10**(power)
    return linearValue

def get_spectral_eff(spectral_eff):
    for i in range(len(spectral_eff_list)-1,-1,-1):
        if(spectral_eff_list[i] <= spectral_eff):
            return spectral_eff_list[i]
    return 0

#receive data from NetSim at the start of the episode
def episode_Start():
    request_value = 0
    msgType = 0
    packed_value = struct.pack(">II",msgType, request_value)

    #send a request value to NetSim
    bytes_Sent = client_socket.send(packed_value)
    
    #recieve SINRs from NetSim
    data = client_socket.recv(bytes_to_receive)

    return data

def NETSIM_interface(gNB_powers):
    
    dummy_array = np.zeros((num_UEs))

    msgType = 1

    # Construct the format string dynamically
    format_string = f'>{len(gNB_powers)}d'
    packed_data = struct.pack(f'>I{format_string[1:]}', msgType, *gNB_powers)

    # packed_data = struct.pack('>Iddd', msgType, gNB_powers[0], gNB_powers[1], gNB_powers[2])
    
    try: 
        #sending a header value indicating that NetSim should recieve the gNB powers
        client_socket.send(packed_data)
        # print("Sent gnb powers ", *gNB_powers)
    except:
        return 0,dummy_array,dummy_array,0
    
    try:
        # receive an acknowledgement message from NetSim
        data = client_socket.recv(4)
        # print("Received value acknowledgement")
    except:
        return 0,dummy_array,dummy_array,0
        

    unpacked_data = struct.unpack('>I', data)

    msgType = 0
    request_value = 2
    packed_value = struct.pack(">II", msgType, request_value)

    try:
        #sending request to NetSim to send over new state and the reward
        client_socket.send(packed_value)
        # print("Sent the request")
    except:
        return 0,dummy_array,dummy_array,0
    

    try:
        #receive the requested data
        data = client_socket.recv(bytes_to_receive)
    except:
        return 0,dummy_array,dummy_array,0
    
    format_string = f'>{2*num_UEs+1}d'
    unpacked_data = struct.unpack(f'>{format_string[1:]}', data)
    new_snirs = [unpacked_data[i] for i in range(num_UEs)]
    new_throughputs = [unpacked_data[i] for i in range(num_UEs,2*num_UEs)]
    
    # print("Received sinrs", new_snirs)

    return 1, new_snirs, new_throughputs, unpacked_data[2*num_UEs]

try:
    for episode in range(num_episodes):

        total_reward = 0  # Initialize total reward for this episode
        
        while True:
            try:
                #establishing connection with NetSim at the start of the episode
                server_address = (socket.gethostbyname(socket.gethostname()), port)
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect(server_address)
                print("Connected to the server")
                break
            except:
                continue
        
        #receive data at the start of the episode
        data = episode_Start()
        

        format_string = f'>{2*num_UEs+1}d'
        unpacked_data = struct.unpack(f'>{format_string[1:]}', data)

        # sinr_values = [unpacked_data[i] for i in range(num_UEs)]
        sinr_values = unpacked_data[0:num_UEs]
        # throughputs_NETSIM = unpacked_data[i] for i in range(num_UEs,2*num_UEs)]
        throughputs_NETSIM = unpacked_data[num_UEs:2*num_UEs]
        # print("Starting of episode, sending values ", sinr_values)

        state = get_state(sinr_values)

        # Reset gNB powers to initial state at the start of each episode
        current_gNB_powers = [40 for _ in range(num_gNBs)]
        steps_per_episode = 500

        for t in range(steps_per_episode):  # Limit steps in each episode

            # print("Start of iteration ", t)
            # print("=========================================================")


            if np.random.rand() < epsilon:
                action_index = np.random.randint(num_actions)  # Explore
            else:
                action_index = np.argmax(q_table[state])  # Exploit

            # Update current_gNB_powers based on the action taken
            delta_p = action_to_power_adjustments(action_index)
            current_gNB_powers = [p + dp for p, dp in zip(current_gNB_powers, delta_p)]

            # Restrict gNB powers between 27 to 46 dBm
            current_gNB_powers = [max(27, min(46, p)) for p in current_gNB_powers]
            
            #recieve next state and reward from NetSim
            flag, next_sinr_values, throughputs_NETSIM, reward = NETSIM_interface(current_gNB_powers)
            


            if flag == 0:
                # next_sinr_values = sinr_values
                continue

            next_state = get_state(next_sinr_values)
            total_reward += reward

            # Q-learning update
            q_table[state, action_index] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action_index])

            # for t in range(steps_per_episode):  # Limit steps in each episode

            #     if np.random.rand() < epsilon:
            #         action_index = np.random.randint(num_actions)  # Explore
            #     else:
            #         action_index = np.argmax(q_table[state])  # Exploit

            #     # Update current_gNB_powers based on the action taken
            #     delta_p = action_to_power_adjustments(action_index)
            #     current_gNB_powers = [p + dp for p, dp in zip(current_gNB_powers, delta_p)]

            #     # Restrict gNB powers between 27 to 46 dBm
            #     current_gNB_powers = [max(27, min(46, p)) for p in current_gNB_powers]
                
            #     # Receive next state and reward from NetSim
            #     flag, next_sinr_values, throughputs_NETSIM, reward = NETSIM_interface(current_gNB_powers)
                
            #     if flag == 0:
            #         continue

            #     next_state = get_state(next_sinr_values)
            #     total_reward += reward

            #     # Log the values here
            #     log_all_values([*throughputs_NETSIM, reward, *next_sinr_values, episode, t])

            #     # Q-learning update
            #     q_table[state, action_index] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action_index])

            #     # Logging the throughputs and gNB powers in the last episode
            #     if episode == num_episodes-1:
            #         log_gnb_powers(current_gNB_powers)

            #     state = next_state
            #     sinr_values = next_sinr_values  

            
            # #calculating the throughput values from the received sinrs 
            # sinr_linear = [dB_to_linear(i) for i in sinr_values]
            # spectral_eff = [np.log2(1 + i) for i in sinr_linear]
            # spectral_eff_NETSIM = [get_spectral_eff(i) for i in spectral_eff]
            # throughputs_Mbps = [(BW_MHz * i * ue_divider * dl_ul_ratio * overhead_multiplier * app_by_phy) for i in spectral_eff_NETSIM]

            # log_all_values([throughputs_Mbps[0],throughputs_Mbps[1],throughputs_Mbps[2],throughputs_Mbps[3],throughputs_Mbps[4],throughputs_Mbps[5], reward, next_sinr_values[0],next_sinr_values[1],next_sinr_values[2],next_sinr_values[3],next_sinr_values[4],next_sinr_values[5], episode, t])
            
            #logging the throughputs and gNB powers in the last episode
            if episode == num_episodes-1:
                # log_sinr_values(throughputs_Mbps)
                log_gnb_powers(current_gNB_powers)


            state = next_state
            sinr_values = next_sinr_values  
            print(f"Episode {episode}, Time_Step {t}")
            # print("================================================")
            # print("Iteration end ", t)

        total_rewards_per_episode.append(total_reward/steps_per_episode)

        # if episode % 1 == 0:

        if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
        print("===============================================================================")
        print("===============================================================================")
        print("===============================================================================")
        print(f"Episode {episode}: Completed, Reward: {total_reward/steps_per_episode}")
        print("===============================================================================")
        print("===============================================================================")
        print("===============================================================================")
        client_socket.close()

except KeyboardInterrupt:
        print("Loop terminated by user.") #If you 

#plotting the average rewards in each episode
plt.plot(total_rewards_per_episode)
plt.title("Average rewards per episode")
plt.xlabel("Episodes")
plt.ylabel("Average Sum Throuhghput(Mbps)")

plot_filename = os.path.join(path_plots, "Sum_Throughputs.png")
plt.savefig(plot_filename)
plt.show()

log_filename = os.path.join(path_logs, "gnb_powers_log.csv")
gnb_DF = pd.read_csv(log_filename)

# log_filename = os.path.join(path_logs, "sinr_values_log.csv")
# sinr_DF = pd.read_csv(log_filename)

# Delete the last row from the DataFrame
gnb_DF.drop(gnb_DF.tail(1).index, inplace=True)
# sinr_DF.drop(sinr_DF.tail(1).index, inplace=True)

gnb1 = gnb_DF["gNB_Power_1"]
gnb2 = gnb_DF["gNB_Power_2"]
gnb3 = gnb_DF["gNB_Power_3"]


gnb_powers_plotting = [gnb_DF[f"gNB_Power_{i+1}"] for i in range(num_gNBs)]
labels = [f"gNB{i+1}" for i in range(num_gNBs)]

# Create subplots
plt.figure(figsize=(10, 18))


# Subplot for gNB1
plt.subplot(3, 1, 1)
plt.plot(gnb1, label='gNB1', color='blue')
plt.title("gNB Powers vs. Time. Optimal Policy")
# plt.xlabel("Iterations")
plt.ylabel("gNB1 Power (dBm)")
plt.legend()

# Subplot for gNB2
plt.subplot(3, 1, 2)
plt.plot(gnb2, label='gNB2', color='green')
# plt.title("gNB2 Power vs. Time. Optimal Policy")
# plt.xlabel("Iterations")
plt.ylabel("gNB2 Power (dBm)")
plt.legend()

# Subplot for gNB3
plt.subplot(3, 1, 3)
plt.plot(gnb3, label='gNB3', color='red')
# plt.title("gNB3 Power vs. Time. Optimal Policy")
plt.xlabel("Iterations")
plt.ylabel("gNB3 Power (dBm)")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
# set the spacing between subplots
plt.subplots_adjust(left=0.062,
                    bottom=0.071, 
                    right=0.985, 
                    top=0.948, 
                    wspace=0.207, 
                    hspace=0.217)

for i in range(len(gnb_powers_plotting)):
    plt.subplot(len(gnb_powers_plotting), 1, i + 1)
    plt.plot(gnb_powers_plotting[i], label=labels[i])
    if i == 0:
        plt.title("gNB Powers vs. Time. Optimal Policy")
    plt.ylabel(f"{labels[i]} Power (dBm)")
    plt.legend()
    if i == len(gnb_powers_plotting) - 1:
        plt.xlabel("Iterations")

plt.tight_layout()
# Save the plot
plot_filename = os.path.join(path_plots, "gNB_Powers.png")
plt.savefig(plot_filename)
plt.show()

# for i in range(num_UEs):
#     #plotting the UE throughputs in separate plots
#     plt.figure(figsize=(10,6))
#     plt.grid(True)
#     plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
#     plt.xlabel("Iterations")
#     plt.ylabel("Individual UE throughput (Mbps)")
#     plt.plot(sinr_DF[f"UE{i+1}_Throughput"])
#     plot_filename = os.path.join(path_plots, f"Individual_UE_throughput_{i+1}.png")
#     plt.savefig(plot_filename)

# # plt.figure(figsize=(10,6))
# # plt.grid(True)
# # plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
# # plt.xlabel("Iterations")
# # plt.ylabel("Individual UE throughput (Mbps)")
# # plt.plot(sinr_DF["UE2_Throughput"])
# # plot_filename = os.path.join(path_plots, "Individual_UE_throughput_2.png")
# # plt.savefig(plot_filename)

# # plt.figure(figsize=(10,6))
# # plt.grid(True)
# # plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
# # plt.xlabel("Iterations")
# # plt.ylabel("Individual UE throughput (Mbps)")
# # plt.plot(sinr_DF["UE3_Throughput"])
# # plot_filename = os.path.join(path_plots, "Individual_UE_throughput_3.png")
# # plt.savefig(plot_filename)

# # plt.figure(figsize=(10,6))
# # plt.grid(True)
# # plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
# # plt.xlabel("Iterations")
# # plt.ylabel("Individual UE throughput (Mbps)")
# # plt.plot(sinr_DF["UE4_Throughput"])
# # plot_filename = os.path.join(path_plots, "Individual_UE_throughput_4.png")
# # plt.savefig(plot_filename)

# # plt.figure(figsize=(10,6))
# # plt.grid(True)
# # plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
# # plt.xlabel("Iterations")
# # plt.ylabel("Individual UE throughput (Mbps)")
# # plt.plot(sinr_DF["UE5_Throughput"])
# # plot_filename = os.path.join(path_plots, "Individual_UE_throughput_5.png")
# # plt.savefig(plot_filename)

# # plt.figure(figsize=(10,6))
# # plt.grid(True)
# # plt.title("Individual UE throughput (Mbps) vs. time. Optimal Policy")
# # plt.xlabel("Iterations")
# # plt.ylabel("Individual UE throughput (Mbps)")
# # plt.plot(sinr_DF["UE6_Throughput"])
# # plot_filename = os.path.join(path_plots, "Individual_UE_throughput_6.png")
# # plt.savefig(plot_filename)


#printing the total time taken for the RL simulation
print(f"Process finished --- {(time.time()-start_time)/60} minutes ---")


# # Save to file. Convert the Q-table to a DataFrame for easier manipulation and saving
q_table_df = pd.DataFrame(q_table)
policy = [np.max(q_table[i]) for i in range(num_states)]
q_table_df.to_csv("Q_Table_interfacing.csv")
policy_df = pd.DataFrame(policy)
policy_df.to_csv("optimal_policy.csv")