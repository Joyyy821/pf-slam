# File: models.py
# Author: Yang Jiao
# Last update: March 7, 2023
# --------------------------
# Include the definition of a class Robot, which implements
# the particle filter SLAM and texture mapping.

import numpy as np
from matplotlib import pyplot as plt

from sensors import LiDAR, IMU, Encoder, Camera
from pr2_utils import mapCorrelation, tic, toc

class Robot:
    def __init__(self, dataset, mode=0, N_pt=10):
        # mode:
        # 0 - dead reckoning
        # 1 - prediction only
        # 2 - particle filter
        # 3 - texture mapping
        self.mode = mode
        self.dataset = dataset
        if mode <= 2:
            self.lidar = LiDAR(dataset)
            self.li_ds_rate = 5  # Downsampling lidar data
            self.encoder = Encoder(dataset)
            self.imu = IMU(dataset)
            self.init_pose = np.eye(4)
            if mode >= 1:
                self.initParticles(N_pt)
                self.state_stamps = np.zeros(int(self.lidar.N/self.li_ds_rate))
            print("Initialize an object instance with %d particles."%N_pt)
            self.initMap()
        else:
            data = np.load("./data/SLAM%d.npz"%self.dataset)
            self.state_stamps = data["stamps"][:-1]
            self.states = data["trajectory"][:-1, :]
            self.camera = Camera(dataset)
            self.initMap(data["map_frames"][:,:,-1])
        self.alignSensors()


    def initMap(self, final_map=[]):
        MAP = {}
        MAP['res']   = 0.05  #meters
        MAP['xmin']  = -10  #meters
        MAP['ymin']  = -10
        MAP['xmax']  =  30
        MAP['ymax']  =  30 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        if self.mode <= 2:
            MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']))
        else:
            final_map = np.transpose(np.flip(final_map, axis=0))*255
            final_map = final_map.astype(int)
            final_map = final_map.reshape((final_map.shape[0], final_map.shape[1], 1))
            MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey'], 3))
            MAP['map'][:, :, :] = final_map
        self.MAP = MAP
        self.x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
        self.y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
        if self.mode <= 2:
            self.map_frames = np.zeros((MAP['sizex'],MAP['sizey'], int(self.lidar.N/100)+1))
        else:
            self.map_frames = np.zeros((MAP['sizex'],MAP['sizey'], 3, int(self.camera.N_rgb/50)+1))
        # show map
        # plt.figure()
        # self.showMap()
    
    def initParticles(self, N):
        self.N_particle = N
        self.mu = np.zeros((self.imu.N, 3, self.N_particle))
        self.alpha = np.zeros((int(self.lidar.N/self.li_ds_rate), \
                              self.N_particle))
        self.alpha[0, :] = 1 / self.N_particle
        self.sigma_v = np.sqrt(0.1)
        if self.dataset == 20:
            w = 0.02
        elif self.dataset ==21:
            w = 0.05
        self.sigma_w = np.sqrt(w)
        self.x_range = np.arange(-0.2,0.2+0.05,0.05)
        self.y_range = np.arange(-0.2,0.2+0.05,0.05)
        self.yaw_perturb = 0.005*4
        self.yaw_pert_res = 0.005
        # an indicator variable
        self.start_update = False

    def predict(self, vt, i):
        mu_t = self.mu[i-1, :, :]  # 3 x N_particle
        tao_t = self.imu.stamps[i] - self.imu.stamps[i-1]
        wt = self.imu.yaw[i]
        e_v, e_w = self.drawGaussianSample(self.N_particle)
        vt = vt + e_v
        wt = wt + e_w
        self.mu[i, :, :] = self.motionModel(mu_t, tao_t, vt, wt)

    def update(self, i_state, i_lidar):
        if not self.start_update:
            self.i_a0 = int(i_lidar/self.li_ds_rate)
            self.start_update = True
        i_a = int(i_lidar/self.li_ds_rate) - self.i_a0
        alpha_i = self.alpha[i_a, :]
        Corr, mu_i = self.getCorrelation(i_state, i_lidar)
        alpha_next = Corr * alpha_i
        self.alpha[i_a+1, :] = alpha_next / np.sum(alpha_next)
        self.mu[i_state, :, :] = mu_i
        # Store the current state with largest weight before resampling
        i_particle = np.argmax(self.alpha[i_a+1, :])
        self.states[i_a+1, :] = self.mu[i_state, :, i_particle]
        self.state_stamps[i_a+1] = self.imu.stamps[i_lidar]
        self.checkForResample(i_state, i_a+1)
        return self.states[i_a+1, :]

    def checkForResample(self, t_mu, t_alpha):
        alpha_t = self.alpha[t_alpha, :]
        N_eff = 1/np.sum(alpha_t**2)
        if N_eff <= self.N_particle/10:
            new_mu = np.zeros((3, self.N_particle))
            print("Resampling at step ", self.imu.stamps[t_mu]-self.imu.stamps[0])
            for i in range(self.N_particle):
                i_mu = np.random.choice(np.arange(self.N_particle), p=alpha_t)
                new_mu[:, i] = self.mu[t_mu, :, i_mu]
            self.mu[t_mu, :, :] = new_mu
            self.alpha[t_alpha, :] = 1 / self.N_particle


    def getCorrelation(self, i_state, i_lidar):
        mu_i = self.mu[i_state, :, :] # 3 x N_particle
        new_mu = np.zeros(mu_i.shape)
        self.get01map()
        Corr = np.zeros(self.N_particle)
        for i in range(self.N_particle):
            Corr[i], new_mu[:, i] = self.getCorrFor1pt(mu_i[:, i], i_lidar)
        return Corr, new_mu
        
    
    def getCorrFor1pt(self, s, i_lidar):
        max_corr = 1
        new_s = s
        yaw_range = np.arange(s[2]-self.yaw_perturb, s[2]+self.yaw_perturb+self.yaw_pert_res, \
                              self.yaw_pert_res)
        N_range = len(yaw_range)
        corr_all = np.zeros((self.x_range.shape[0],
                             self.y_range.shape[0],
                             N_range))
        for i in range(N_range):
            s_yaw = yaw_range[i]
            s_per = np.array([s[0], s[1], s_yaw])
            T = self.state2pose(s_per)
            scan_pts = self.lidar.getScan(T, i_lidar)
            # Perturbation on x, y
            corr_all[:, :, i] = mapCorrelation(self.occup_map,self.x_im, self.y_im, scan_pts, \
                                    self.x_range, self.y_range)
        inds = np.argwhere(corr_all == np.amax(corr_all))
        N_arg = inds.shape[0]
        rand_ind = np.random.choice(np.arange(N_arg))
        ind = inds[rand_ind]
        max_corr = corr_all[ind[0], ind[1], ind[2]]
        new_s = np.array([s[0]+self.x_range[ind[0]], s[1]+self.y_range[ind[1]], yaw_range[ind[2]]])
        return max_corr, new_s

    def deadRecon(self, vt, i):
        xt = self.states[i-1, :]
        tao_t = self.imu.stamps[i] - self.imu.stamps[i-1]
        wt = self.imu.yaw[i]
        self.states[i] = self.motionModel(xt, tao_t, vt, wt)
        
    def SLAM(self):
        # Align the IMU and encoder data, and calculate the trajectory by motion model
        x0 = self.pose2state(self.init_pose)
        if self.mode <= 1:
            self.states = np.zeros((self.imu.N, 3))
        else:
            self.states = np.zeros((int(self.lidar.N/self.li_ds_rate), 3))
            self.state_stamps[0] = self.lidar.stamps[0]
        self.states[0, :] = x0
        if self.mode == 1 or self.mode == 2:
            self.mu[0, 0, :] = x0[0]
            self.mu[0, 1, :] = x0[1]
            self.mu[0, 2, :] = x0[2]
        # encoder, imu, lidar
        ei, ii, li = 0, 0, 0
        for i in range(self.N_data):
            if self.stamp_types[i] == 1: # Encoder data
                vt = self.encoder.v[ei]
                ei += 1
            elif self.stamp_types[i] == 2: # IMU data
                if not ii:
                    # skip the first IMU data (need tao_t)
                    ii += 1
                    continue
                if self.mode == 0 or self.mode == 1:
                    self.deadRecon(vt, ii)
                if self.mode == 1 or self.mode == 2:
                    self.predict(vt, ii)
                ii += 1
            elif self.stamp_types[i] == 3: # LiDAR data
                if not li % self.li_ds_rate:
                    if not ii:
                        i_state = ii
                    else:
                        i_state = ii-1
                    # if ii == self.imu.N or (np.all(np.abs(self.states[ii, :] < 1e-4)) and ii):
                    #     i_state = ii-1
                    #     # c_state = self.states[ii-1, :]
                    # else:
                    #     i_state = ii
                    if self.mode == 2:
                        if ii >= 2:
                            c_state = self.update(i_state, li)
                        else:
                            c_state = x0
                    else:
                        c_state = self.states[i_state, :]
                    c_pose = self.state2pose(c_state)
                    self.lidar.updateMap(self.MAP, c_pose, li)
                    if not li % 100:
                        print("Time: ", self.lidar.stamps[li]-self.lidar.stamps[0], \
                              " | Robot state: ", c_state)
                        # Store the map frame
                        self.showMap()
                        # self.getDisplayMap()
                        self.map_frames[:, :, int(li/100)] = self.disp_map
                li += 1
            
        # Save the results from particle filter SLAM (mode=2)
        if self.mode == 2:
            outfile = "./data/SLAM%d.npz"%self.dataset
            np.savez(outfile, trajectory=self.states, \
                     particles=self.mu, \
                     map_frames=self.map_frames, \
                     stamps=self.state_stamps)
            print("Filtering results saved to ", outfile)
            # elif self.mode == 3 and self.stamp_types[i] == 4: # Disparity data
            #     di += 1
            # elif self.mode == 3 and self.stamp_types[i] == 5: # RGB data
            #     ci += 1
            #     if not li:
            #         c_pose = self.state2pose(x0)
            #     self.camera.textureFrom1Img(self.MAP, c_pose, ci, di)
            #     # TODO show the map

    def textureMapping(self):
        si, di, ci = 0, 0, 0 # State, disparity, camera rgb indices
        for i in range(self.N_data):
            if self.stamp_types[i] == 1:  # Robot state
                # if not di or not ci:
                #     # Wait until getting all image stamps
                #     si += 1
                #     continue
                # c_pose = self.state2pose(self.states[si])
                # self.camera.textureFrom1Img(self.MAP, c_pose, ci, di)
                # self.map_frames[:, :, :, si] = self.MAP['map']
                # if not si % 50:
                #     self.showMap()
                si += 1
            elif self.stamp_types[i] == 2:  # disparity
                di += 1
            elif self.stamp_types[i] == 3:  # rgb
                ci += 1
                if not di or not si:
                    # Wait until getting all image stamps
                    # ci += 1
                    continue
                c_pose = self.state2pose(self.states[si-1])
                self.camera.textureFrom1Img(self.MAP, c_pose, ci, di)
                if not ci % 50:
                    self.map_frames[:, :, :, int(ci/50)] = self.MAP['map']
                    self.showMap()
        # Save all map frames
        outfile = "./data/Texture%d.npz"%self.dataset
        np.savez(outfile, map_frames=self.map_frames, stamps=self.state_stamps)
        print("Texture mapping results saved to ", outfile)


    def alignSensors(self):
        # For SLAM: Encoder: 1, IMU: 2, LiDAR: 3
        # For texture mapping: Robot states: 1, Camera diparity: 2, Camera RGB: 3
        if self.mode <= 2:
            self.N_data = self.encoder.N + self.imu.N + self.lidar.N
            all_stamps = np.append(np.append(self.encoder.stamps, self.imu.stamps), \
                                self.lidar.stamps)
            N1 = self.encoder.N
            N2 = self.encoder.N+self.imu.N
        else:
            self.N_data = self.state_stamps.shape[0] + self.camera.N_disp + self.camera.N_rgb
            all_stamps = np.append(np.append(self.state_stamps, self.camera.disp_stamps), \
                                self.camera.rgb_stamps)
            N1 = self.state_stamps.shape[0]
            N2 = self.state_stamps.shape[0]+self.camera.N_disp
        stamp_types = np.ones(self.N_data)
        stamp_types[N1:] = 2
        stamp_types[N2:] = 3
        stamp_idxs = np.argsort(all_stamps)
        self.stamps = all_stamps[stamp_idxs]
        self.stamp_types = stamp_types[stamp_idxs]

    def unitTestCorrelation(self):
        s = np.array([0., 0., 0.])
        c_pose = self.state2pose(s)
        self.lidar.updateMap(self.MAP, c_pose, 0)
        self.get01map()
        corr, new_s = self.getCorrFor1pt(s, 0)
        self.showMap()
        print("corr: ", corr)
        print("new_s: ", new_s)
        plt.ioff()
        plt.show()


    ## Helper functions

    def getDisplayMap(self):
        self.disp_map = np.ones(self.MAP['map'].shape) * 0.5
        indFree = (self.MAP['map'] < -1e-3)
        indOccup = (self.MAP['map'] > 1e-3)
        self.disp_map[indFree] = 1
        self.disp_map[indOccup] = 0
        self.disp_map = np.flip(self.disp_map.T, axis=0)

    def get01map(self):
        self.occup_map = np.zeros(self.MAP['map'].shape)
        indOccup = (self.MAP['map'] > 1e-3)
        self.occup_map[indOccup] = 1

    def showMap(self):
        if self.mode <= 2:
            self.getDisplayMap()  # update the disp_map attribute
            plt.imshow(self.disp_map,cmap="gray",vmin=0, vmax=1)
            plt.title("Occupancy grid map")
        else:
            plt.imshow(np.flip(np.transpose(normalize(self.MAP['map']), axes=(1, 0, 2)), axis=0))
            plt.title("Texture map")
        plt.draw()
        plt.pause(0.01)

    def showParticles(self):
        plt.ioff()
        plt.figure()
        for i in range(self.N_particle):
            plt.plot(self.mu[:, 0, i], self.mu[:, 1, i])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Particle trajectories")

    def motionModel(self, xt, tao_t, vt, wt):
        # xt - 3 x N_particle
        # tao_t - scalar
        # vt - N_particle
        # wt - N_particle
        if np.isscalar(vt):
            x_next = xt + tao_t * np.array([vt*np.cos(xt[2]), vt*np.sin(xt[2]), wt])
        else:
            x_next = xt + tao_t * np.vstack((vt*np.cos(xt[2,:]), vt*np.sin(xt[2,:]), wt))
        return x_next

    def state2pose(self, state):
        T = np.eye(4)
        yaw = state[2]
        T[0:2, 0:2] = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                [np.sin(yaw), np.cos(yaw)]])
        T[0, 3], T[1, 3] = state[0], state[1]
        return T

    def pose2state(self, T):
        px, py = T[0, 3], T[1, 3]
        yaw = self.logMap(T[0:3, 0:3])
        return np.array([px, py, yaw])

    def logMap(self, R):
        #  Only returns the yaw angle (the only rotation used in 2D)
        tolerance = 1e-3
        if np.linalg.norm(R - np.eye(3)) < tolerance:
            # Treat R as I
            theta = 0
        elif np.abs(np.trace(R) - (-1)) < tolerance:
            # check if it is a rotation along z
            if np.abs(R[2, 2] - 1) < tolerance:
                theta = np.pi
            else:
                print("Warning: got rotation matrix: ")
                print(R)
                print("which is not along z axis.")
        else:
            theta = np.arccos(1/2*(np.trace(R)-1))
            omega_hat = 1/(2*np.sin(theta))*(R - R.T)
            omega_z = omega_hat[1, 0]
            theta = theta * omega_z
        return theta

    def drawGaussianSample(self, n_sample=1):
        epsilon_v = np.random.normal(0, self.sigma_v, n_sample)
        epsilon_w = np.random.normal(0, self.sigma_w, n_sample)
        return epsilon_v,epsilon_w

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

def drawMaps():
    file = "./data/Texture21.npz"
    data = np.load(file)
    # maps = data["map_frames"]
    maps = data["map_frames"][:, :, :, 1:]
    N = maps.shape[-1]
    N_frames = 8
    N_cols = int(N_frames / 2)
    N_rows = int(N_frames / N_cols)
    plt.figure()
    for i in range(N_frames):
        plt.subplot(N_rows, N_cols, i+1)
        c_map = maps[:, :, :, int(i*N/N_frames)]
        # plt.imshow(c_map, cmap="gray", vmin=0, vmax=1)
        plt.imshow(np.flip(np.transpose(normalize(c_map), axes=(1, 0, 2)), axis=0))
        # plt.title("Map frame "+str(i+1))
        plt.title("Texture frame "+str(i+1))

def drawTrajectory():
    file = "./data/SLAM21.npz"
    data = np.load(file)
    states = data["trajectory"][:-1, :]
    N = states.shape[0]
    N_frames = 8
    N_cols = int(N_frames / 2)
    N_rows = int(N_frames / N_cols)
    plt.figure()
    for i in range(N_frames):
        plt.subplot(N_rows, N_cols, i+1)
        c_states = states[:int(i*N/N_frames), :]
        plt.plot(c_states[:, 0], c_states[:, 1])
        plt.xlim([-10, 30])
        plt.ylim([-10, 30])
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("Trajectory frame "+str(i+1))

def drawParticles():
    file = "./data/SLAM20.npz"
    data = np.load(file)
    states = data["particles"][:-1, :, :]
    N = states.shape[0]
    N_frames = 8
    N_cols = int(N_frames / 2)
    N_rows = int(N_frames / N_cols)
    plt.figure()
    for i in range(N_frames):
        plt.subplot(N_rows, N_cols, i+1)
        for j in range(states.shape[-1]):
            plt.plot(states[:int(i*N/N_frames), 0, j], states[:int(i*N/N_frames), 1, j])
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.xlim([-10, 30])
        plt.ylim([-10, 30])
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.title("Particles frame "+str(i+1))

if __name__ =="__main__":
    drawMaps()
    # drawTrajectory()
    # drawParticles()

    # # Get the result from dead reckoning
    # N_pt = 200
    # rob1 = Robot(dataset=20, mode=2, N_pt=N_pt)

    # # rob1.unitTestCorrelation()

    # # rob1.camera.textureFrom1Img(rob1.MAP, np.eye(4), 1, 1)
    # # plt.imshow(np.flip(np.transpose(normalize(rob1.MAP['map']), axes=(1, 0, 2)), axis=0))

    # # rob1.textureMapping()

    # ts = tic()
    # rob1.SLAM()
    # toc(ts, "Particle filter with "+str(N_pt)+" particles")

    # # plot map
    # # plt.figure()
    # # rob1.showMap()

    # # Plot the trajectory
    # rob1.showParticles()

    # plt.ioff()
    # plt.figure()
    # plt.plot(rob1.states[:,0], rob1.states[:,1])
    # plt.title("Robot state x-y trajectory")
    # plt.figure()
    # plt.plot(rob1.state_stamps-rob1.state_stamps[0], rob1.states[:,2])
    # plt.title("Robot state yaw angle")

    plt.show()
