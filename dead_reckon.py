# dead reckoning
import numpy as np
from pr2_utils import *
import matplotlib.pyplot as plt

def motion_model():
    # parameters
    res = 4096
    dia_left = 0.623479
    dia_right = 0.622806
    
    file_enc = '/Users/vb/Desktop/ece276a/ECE276A_PR2/code/data/sensor_data/encoder.csv'
    time_enc, data_enc = read_data_from_csv(file_enc)
    file_fog = '/Users/vb/Desktop/ece276a/ECE276A_PR2/code/data/sensor_data/fog.csv'
    time_fog, data_fog = read_data_from_csv(file_fog)
    
    x = np.zeros((len(time_enc), 3))
    # t_mean = np.zeros(len(time_enc)-1)
    index = 0
    theta = 0
    for t in range(len(time_enc)-1):
#         del_t_enc = time_enc[t]-time_enc[t-1]
        x_left = np.pi*dia_left*(data_enc[t+1][0]-data_enc[t][0])/res
        x_right = np.pi*dia_right*(data_enc[t+1][1]-data_enc[t][1])/res
        x_mean = (x_left + x_right)/2
        
        # counter = 0
        for counter in range(20):
            theta += data_fog[t+1+index+counter][2]
            if time_fog[t+1+index+counter] >= time_enc[t+1]:
                break
#             theta += data_fog[t+index+counter][2]    
        index += counter
#         del_t_fog = time_fog[t+index]-time_fog[t]
#         print(time_enc[t], time_fog[t+index])
#         print(del_t_enc, del_t_fog)    
        
#         t_mean[t] = (time_enc[t] + time_fog[t+index])/2            
        
        temp = np.array([x_mean*np.cos(theta), x_mean*np.sin(theta), theta])
        x[t+1] = x[t] + temp
    
    return x   

mm = motion_model()


plt.plot(mm[:,0], mm[:,1])
plt.gca().set_aspect("equal")
plt.show()