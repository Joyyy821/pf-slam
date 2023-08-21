# File: sensors.py
# Author: Yang Jiao
# Last update: March 7, 2023
# --------------------------
# Implement the class representation of all 4 kinds of sensors 
# for the differential-drive robot.


import numpy as np
import cv2
from matplotlib import pyplot as plt

from pr2_utils import bresenham2D

class LiDAR:
  def __init__(self, dataset):
    # Transformation to body (meter)
    self.tf_body = np.array([[1, 0, 0, 0.13323],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.51435],
                       [0, 0, 0, 1]])
    with np.load("./data/Hokuyo%d.npz"%dataset) as data:
      self.angle_min = data["angle_min"] # start angle of the scan [rad]
      self.angle_max = data["angle_max"] # end angle of the scan [rad]
      self.angle_increment = data["angle_increment"] # angular distance between measurements [rad]
      self.range_min = data["range_min"] # minimum range value [m]
      self.range_max = data["range_max"] # maximum range value [m]
      self.ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
      self.stamps = data["time_stamps"]  # acquisition times of the lidar scans
    self.N_beam = self.ranges.shape[0]
    self.N = len(self.stamps)
    self.polar2cartesian()
    # For mapping
    self.lambda_max = 2*np.log(4)
    self.lambda_min = -2*np.log(4)

  def getScan(self, T_wb, range_i):
    # Get valid range measurement
    c_ranges = self.ranges[:, range_i]
    indValid = np.logical_and((c_ranges < 30),(c_ranges> 0.1))

    c_x, c_y = self.x[indValid, range_i], self.y[indValid, range_i]
    # Pose transformation from laser frame to world frame
    self.T_wl = np.matmul(T_wb, self.tf_body)
    pts_li_frame = np.vstack((c_x, c_y, np.zeros(c_x.shape), np.ones(c_x.shape)))
    pts_w_frame = np.matmul(self.T_wl, pts_li_frame)
    return pts_w_frame[0:2, :]

  def updateMap(self, MAP, T_wb, range_i):
    scan_pts = self.getScan(T_wb, range_i)
    # (x, y) from lidar pose
    lx, ly = self.T_wl[0, 3], self.T_wl[1, 3]
    for j in range(scan_pts.shape[1]):
      xw, yw = scan_pts[0,j], scan_pts[1,j]
      # convert from meters to cells
      xis = np.round((np.array([lx, xw]) - MAP['xmin']) / MAP['res'] ).astype(np.int16)
      yis = np.round((np.array([ly, yw]) - MAP['ymin']) / MAP['res'] ).astype(np.int16)
      x_path, y_path = bresenham2D(xis[0], yis[0], xis[1], yis[1]).astype(np.int16)
      
      # build an arbitrary map 
      indGood = np.logical_and(np.logical_and(np.logical_and((x_path > 1), 
                                                             (y_path > 1)), 
                                                             (x_path < MAP['sizex'])), 
                                                             (y_path < MAP['sizey']))
      # Avoid overconfident estimation
      # Empty cells
      MAP['map'][x_path[indGood], y_path[indGood]] = np.maximum(self.lambda_min, \
                          MAP['map'][x_path[indGood], y_path[indGood]] - np.log(4))
      # Obstacle edges
      if indGood[-1]:
        MAP['map'][xis[1], yis[1]] = np.min([self.lambda_max, MAP['map'][xis[1], yis[1]]+2*np.log(4)])


  def polar2cartesian(self):
    self.angles = np.arange(self.angle_min, 
                            self.angle_max+self.angle_increment-1e-4,
                            self.angle_increment) # *np.pi/180.0
    # xy position in the sensor frame
    self.x = self.ranges*np.reshape(np.cos(self.angles), (self.N_beam, 1))
    self.y = self.ranges*np.reshape(np.sin(self.angles), (self.N_beam, 1))


class IMU:
  def __init__(self, dataset):
    with np.load("./data/Imu%d.npz"%dataset) as data:
      self.angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
      self.linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
      self.stamps = data["time_stamps"]  # acquisition times of the imu measurements
    self.yaw = self.angular_velocity[2, :]
    self.N = len(self.stamps)


class Encoder:
  def __init__(self, dataset):
    with np.load("./data/Encoders%d.npz"%dataset) as data:
      self.counts = data["counts"] # 4 x n encoder counts
      self.stamps = data["time_stamps"] # encoder time stamps
    self.l = 0.254*np.pi/360 # m/tic
    self.freq = 40 # Hz
    self.getV()
    self.N = len(self.stamps)

  def getV(self):
    self.v = np.mean(self.counts, axis=0)*self.l * self.freq


class Camera:
  def __init__(self, dataset):
    # Read time stamps
    with np.load("./data/Kinect%d.npz"%dataset) as data:
      self.disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
      self.rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    self.N_disp = len(self.disp_stamps)
    self.N_rgb = len(self.rgb_stamps)
    # Read Images
    self.path_disp = "./data/dataRGBD/Disparity"+str(dataset)+"/disparity"+str(dataset)+"_"
    self.path_rgb = "./data/dataRGBD/RGB"+str(dataset)+"/rgb"+str(dataset)+"_"
    # Transformation
    self.getTF()
    self.floor_th = [-0.01, 0.01]
    
  def getTF(self):
    # Camera regular frame to optical frame
    self.tf_ro = np.array([[ 0,  0, 1, 0],
                           [-1,  0, 0, 0],
                           [ 0, -1, 0, 0],
                           [ 0,  0, 0, 1]])
    # Car body frame to camera regular frame
    # pitch = 0.36 # (rad)
    pitch = 18 * np.pi / 180 # (rad)
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    # yaw = 0.021
    # R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
    #                 [np.sin(yaw), np.cos(yaw), 0],
    #                 [0, 0, 1]])
    # R = np.matmul(R_y, R_z)
    R = R_y
    self.tf_br = np.eye(4)
    self.tf_br[0:3, 0:3] = R
    self.tf_br[0:3, 3] = np.array([0.16766, 0, 0.38001])
    # Cascaded transformation
    self.tf_body = np.matmul(self.tf_br, self.tf_ro)
    # Intrinsic projection
    self.fx = 585.05108211
    self.fy = 585.05108211
    self.cx = 315.83800193
    self.cy = 242.94140713
    # K = np.array([[585.05108211, 0, 242.94140713],
    #               [0, 585.05108211, 315.83800193],
    #               [0, 0, 1]])
    # self.K_inv = np.linalg.inv(K)
  
  def readImg(self, i_rgb, i_disp):
    # load RGBD image
    imd = cv2.imread(self.path_disp+str(i_disp)+'.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(self.path_rgb+str(i_rgb)+'.png')[...,::-1] # (480 x 640 x 3)
    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)
    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    # get 3D coordinates
    x = (u-self.cx) / self.fx * z
    y = (v-self.cy) / self.fy * z
    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/self.fx)
    rgbv = np.round((v * 526.37 + 16662.0)/self.fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    valid = valid&(z>=0.1)&(z<10)
    # pixels in camera optical frame (in physical unit with depth)
    p_o = np.vstack((x[valid], y[valid], z[valid]))
    return imc, p_o, rgbv[valid], rgbu[valid]

  def textureFrom1Img(self, MAP, T_wb, i_rgb, i_disp):
    T_wo = np.matmul(T_wb, self.tf_body)  # world to camera optical frame
    imc, p_o, rgbv, rgbu = self.readImg(i_rgb, i_disp)
    p_o = np.vstack((p_o, np.ones((1, p_o.shape[1]))))
    p_w = np.matmul(T_wo, p_o)
    ind_floor = (p_w[2,:] >= self.floor_th[0]) & (p_w[2,:] <= self.floor_th[1])
    # update global texture map
    x_floor = np.round((p_w[0, ind_floor] - MAP['xmin']) / MAP['res'] ).astype(np.int16)
    y_floor = np.round((p_w[1, ind_floor] - MAP['ymin']) / MAP['res'] ).astype(np.int16)
    # print("map: ", MAP['map'].shape)
    # print("imc: ", imc.shape)
    MAP['map'][x_floor, y_floor, :] = imc[rgbv[ind_floor].astype(int),rgbu[ind_floor].astype(int), :]


def myplot(x, y):
  plt.figure()
  plt.plot(x, y)
