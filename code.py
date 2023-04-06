import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import *

## read all files

# motion
file_enc = '/Users/vb/Desktop/ece276a/ECE276A_PR2/code/data/sensor_data/encoder.csv'
time_enc, data_enc = read_data_from_csv(file_enc)
file_fog = '/Users/vb/Desktop/ece276a/ECE276A_PR2/code/data/sensor_data/fog.csv'
time_fog, data_fog = read_data_from_csv(file_fog)

# observation
file_ldr = '/Users/vb/Desktop/ece276a/ECE276A_PR2/code/data/sensor_data/lidar.csv'
time_ldr, data_ldr = read_data_from_csv(file_ldr)

## parameters

# encoder
res = 4096
dia_left = 0.623479
dia_right = 0.622806

# lidar
start_ang = -5
end_ang = 185

# lidar to vehicle frame transformation pose
R = np.array([[0.00130201, 0.796097, 0.605167], 
              [0.999999, -0.000419027, -0.00160026], 
              [-0.00102038, 0.605169, -0.796097]]) 
T = np.array([0.8349, -0.0126869, 1.76416, 1])
P_ldr2bdy = np.array([[R[0][0], R[0][1], R[0][2], T[0]],
                      [R[1][0], R[1][1], R[1][2], T[1]],
                      [R[2][0], R[2][1], R[2][2], T[2]],
                      [0, 0, 0, 1]])

# lidar readings per scan
l = len(data_ldr[0])                        


## function to generate particle trajectories

def trajectory_estm(t):
    
    global theta

    x_left = np.pi * dia_left * (data_enc[t+1][0] - data_enc[t][0]) / res
    x_right = np.pi * dia_right * (data_enc[t+1][1] - data_enc[t][1]) / res
    x_mean = (x_left + x_right) / 2
    theta += del_theta[t]
    
    # differential drive model for vehicle
    temp = np.array([x_mean * np.cos(theta), x_mean * np.sin(theta), theta])
    x[t+1] = x[t] + temp

    # particle state change
    del_x = x_mean * np.cos(particle[2] + del_theta[t])
    del_y = x_mean * np.sin(particle[2] + del_theta[t])

    # new particle state
    particle[0] += del_x
    particle[1] += del_y
    particle[2] += del_theta[t]

    # add noise to particles
    particle[0] += np.random.normal(0, abs(np.max(del_x)) / 10, n)
    particle[1] += np.random.normal(0, abs(np.max(del_y)) / 10, n)
    particle[2] += np.random.normal(0, abs(del_theta[t]) / 10, n)
    
    pass


## function to convert lidar readings to world frame

def lidar_world_trans(rad, dat):

    # filter lidar range
    phi = np.radians(np.linspace(start_ang, end_ang, l))
    req_ind = np.logical_and(rad < 75, rad > 2)
    rad = rad[req_ind]
    phi = phi[req_ind]

    # convert polar to cartesian
    x_ldr = rad * np.cos(phi)
    y_ldr = rad * np.sin(phi)

    # convert points to 3D coordinates
    p = np.ones((4, len(phi)))
    p[0] = x_ldr
    p[1] = y_ldr
    p[2] = 0

    # vehicle to world frame transformation pose
    P_bdy2world = np.array([[np.cos(dat[2]), -np.sin(dat[2]), 0, dat[0]],
                            [np.sin(dat[2]),  np.cos(dat[2]), 0, dat[1]],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])       

    # transform end point to world frame
    p = np.dot(P_ldr2bdy, p)
    p = np.dot(P_bdy2world, p)

    return p[0], p[1]


## function to update map

def mapping_update(MAP, px, py, dat):
    
    # convert ray start and end coordinates into cells
    sx = np.ceil((dat[0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    sy = np.ceil((dat[1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    ex = np.ceil((px - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ey = np.ceil((py - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    # implement bresenham2D for ray tracing
    for i in range(len(px)):
        b_pts = bresenham2D(sx, sy, ex[i], ey[i])
        b_x = b_pts[0, :].astype(np.int16)
        b_y = b_pts[1, :].astype(np.int16)

        req_ind = np.logical_and(
            np.logical_and(np.logical_and((b_x > 1), (b_y > 1)), (b_x < MAP['sizex'])), (b_y < MAP['sizey']))

    # update map
    
    # decrease log-odds if cell observed free
        MAP['map'][b_x[req_ind], b_y[req_ind]] -= np.log(4)

    # increase log-odds if cell observed occupied
    for i in range(len(px)):
        if ((ex[i] > 1) and (ex[i] < MAP['sizex']) and (ey[i] > 1) and (ey[i] < MAP['sizey'])):
            MAP['map'][ex[i], ey[i]] += 2*np.log(4)

    # clip range to prevent over-confidence
    MAP['map'] = np.clip(MAP['map'], -10*np.log(4), 10*np.log(4))
    
    return MAP


## function to estimate the robot trajectory via differential drive motion model

def best_particle_estm(rad, dat, wt, MAP, x_im, y_im, x_range, y_range): 
    
    correlation = np.zeros(n)

    for i in range(n):
        # convert lidar points to world frame
        px, py = lidar_world_trans(rad, dat[:, i])
        Y = np.stack((px, py))
        
        # compute map correlation for each particle
        temp = mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range)

        # find best correlation
        correlation[i] = np.max(temp)

    # update particle weights with softmax
    mx = np.max(correlation)
    beta = np.exp(correlation - mx)
    prt = beta / np.sum(beta)
    wt *= prt / np.sum(wt * prt)
    
    # estimate best-matched particle
    ind = np.argmax(wt) 
    best = dat[:, ind]
    
    return best


## function to plot map

def map_plot(MAP, t):

    # recover map pmf from log-odds
    binary_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(np.int)
    binary_wall = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.9).astype(np.int)

    # convert trajectory to map coordinates
    tx = np.ceil((trajectory[:t,0] - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    ty = np.ceil((trajectory[:t,1] - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1

    # plt.imshow(MAP['map'], cmap = 'gray')
    plt.imshow(binary_map, cmap = 'gray')
    plt.imshow(binary_wall, cmap = 'gray')
    plt.plot(ty, tx, color = 'orangered', linewidth = 0.5)
    plt.title("Binary Map")
    plt.xlabel("X grid coordinates")
    plt.ylabel("Y grid coordinates")
    # plt.colorbar()
    plt.show(block=True)

    pass


## function to resample particles

def particle_resampling(dat, wt):

    dat_new = np.zeros((3, n))
    wt_new = np.tile(1 / n, n).reshape(1, n)
    j = 0
    c = wt[0, 0]

    for i in range(n):
        u = np.random.uniform(0, 1/n)
        beta = u + i/n
        while beta > c:
            j += 1
            c += wt[0, j]

        # add to the new set
        dat_new[:, i] = dat[:, j]

    return dat_new, wt_new


## function to initialize map

def init_map(MAP):
        
    # initialize map
    MAP['res']      = 1 # Meters
    MAP['xmin']     = -100 # Meters
    MAP['ymin']     = -1200 # Meters
    MAP['xmax']     = 1300 # Meters
    MAP['ymax']     = 200  # Meters
    MAP['sizex']    = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']    = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map']      = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32) #DATA TYPE: char or int8

    # map correlation arguments
    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  #x-positions of each pixel of map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  #y-positions of each pixel of map

    # 9x9 grid around particle
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res']) 
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])

    return MAP, x_im, y_im, x_range, y_range 

## Particle Filter SLAM execution main thread

# init objects 

# initialize particles
n = 5
particle = np.zeros((3, n))  # mu = [x,y,theta]
alpha = np.ones((1, n))/n  # alpha = 1/N

# vehicle coordinates array [x, y, theta]
x = np.zeros((len(time_enc), 3))

# final particle trajectory
trajectory = np.zeros((len(time_enc), 3))

# init counters and cum-angle
theta = 0
counter = 0

# sync encoder with fog
del_theta = [sum(data_fog[(i - 1) * 10 + 1 : i * 10 + 1, 2]) for i in range(len(time_enc)-1)]

# initialize map
MAP = {}
MAP, x_im, y_im, x_range, y_range = init_map(MAP)

# map generation
for t in range(len(time_enc)-1):

    # predict particle trajectories   
    trajectory_estm(t)
        
    # sync lidar with encoder to update map
    if time_ldr[counter] < time_enc[t]:
        best_particle = best_particle_estm(data_ldr[counter], particle, alpha, MAP, x_im, y_im, x_range, y_range)
        px_world, py_world = lidar_world_trans(data_ldr[counter], best_particle)
        MAP = mapping_update(MAP, px_world, py_world, best_particle)
        counter += 5

    trajectory[t] = best_particle    
    if counter >= len(time_ldr):
        break

    # resample if needed
    n_eff = 1/np.dot(alpha.reshape(1, n), alpha.reshape(n, 1))
    if n_eff < 0.4 * n:
        particle, alpha = particle_resampling(particle, alpha)

    # plot map    
    # if not t % 10000:
    #     map_plot(MAP, counter)
    #     # print('Encoder Counter:', t)
    #     # print('Lidar Counter:', counter)

map_plot(MAP, counter)
      
           
    




        
           
    




