import gym
from gym import spaces
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 

USED_HISTORY_STAMP = 1

SIM_RESOLUTION = .1
HIT_PENALTY = 100
SPEED_REWARD = 1
SPEED_DIFF_REWARD = 1

class vehicle:
  def __init__(self, speed, loc):
    self.location = loc[USED_HISTORY_STAMP - 1]
    self.speed = speed[USED_HISTORY_STAMP - 1]
    self.locT = np.array(loc[:USED_HISTORY_STAMP])
    self.speedT = np.array(speed[:USED_HISTORY_STAMP])

  def action_a(self, aRate):
    self.speed = max(0, self.speed + aRate * SIM_RESOLUTION)
    self.location = self.location + self.speed * SIM_RESOLUTION
    self.locT = np.append(self.locT[1:], [self.location])
    self.speedT = np.append(self.speedT[1:], [self.speed])
  def action_v(self, speed):
    self.speed = speed
    self.location = self.location + speed * SIM_RESOLUTION
    self.locT = np.append(self.locT[1:], [self.location])
    self.speedT = np.append(self.speedT[1:], [self.speed])


class CFEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}

  def __init__(self, rewardFunction):
    super(CFEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    self.leaderFile = 'test_leader_high.csv'
    self.leadSpeedProfile = np.genfromtxt(self.leaderFile, delimiter=',')
    self.initialSpacing = 30
    leaderLoc = [0]
    followerLoc = [-self.initialSpacing]
    for v in self.leadSpeedProfile[1:USED_HISTORY_STAMP]:
      leaderLoc.append(leaderLoc[-1] + v * SIM_RESOLUTION)
      followerLoc.append(followerLoc[-1] + v * SIM_RESOLUTION)
    self.follower = vehicle(self.leadSpeedProfile[:USED_HISTORY_STAMP], followerLoc)
    self.leader = vehicle(self.leadSpeedProfile[:USED_HISTORY_STAMP], leaderLoc)
    self.followerSpeedProfile = self.leadSpeedProfile[:USED_HISTORY_STAMP]
    self.followerSpacing = np.array([self.initialSpacing] * USED_HISTORY_STAMP)
    self.t = USED_HISTORY_STAMP
    self.crash = False

    self.rewardName = rewardFunction
    self.action_space = spaces.Box(low=np.array([-5.]), high=np.array([5.]),
                                  shape=(1,), dtype=np.float64)
    self.observation_space = spaces.Box(low=np.array([0., 0., -np.inf] * USED_HISTORY_STAMP),
                                        high=np.array([np.inf] * USED_HISTORY_STAMP * 3),
                                      shape=(int(USED_HISTORY_STAMP * 3),), dtype=np.float64)


  def reset(self):
    # Reset the state of the environment to an initial state
    self.leadSpeedProfile = np.genfromtxt(self.leaderFile, delimiter=',')
    leaderLoc = [0]
    followerLoc = [-self.initialSpacing]
    for v in self.leadSpeedProfile[1:USED_HISTORY_STAMP]:
      leaderLoc.append(leaderLoc[-1] + v * SIM_RESOLUTION)
      followerLoc.append(followerLoc[-1] + v * SIM_RESOLUTION)
    self.follower = vehicle(self.leadSpeedProfile[:USED_HISTORY_STAMP], followerLoc)
    self.leader = vehicle(self.leadSpeedProfile[:USED_HISTORY_STAMP], leaderLoc)
    self.followerSpeedProfile = self.leadSpeedProfile[:USED_HISTORY_STAMP]
    self.followerSpacing = np.array([self.initialSpacing] * USED_HISTORY_STAMP)
    self.t = USED_HISTORY_STAMP
    self.crash = False

    return np.concatenate((self.leader.speedT, self.follower.speedT,
                          [self.leader.locT[i] - self.follower.locT[i] for i in range(USED_HISTORY_STAMP)]))

  def step(self, action):
    # Execute one time step within the environment
    done = False
    #new_states
    self.follower.action_a(action)
    self.leader.action_v(self.leadSpeedProfile[self.t])


    newSpacing = self.leader.location - self.follower.location
    newStates = np.concatenate((self.leader.speedT, self.follower.speedT,
                          [self.leader.locT[i] - self.follower.locT[i] for i in range(USED_HISTORY_STAMP)]))
    self.followerSpeedProfile = np.append(self.followerSpeedProfile, [self.follower.speed])
    self.followerSpacing = np.append(self.followerSpacing, newSpacing)

    #current obs
    self.t += 1
    observation = newStates
    reward = self.get_reward(action)
    if newSpacing < 0 or self.t >= len(self.leadSpeedProfile):
      done = 1
    else:
      done = 0
    return observation, reward, done, {}

  def get_reward(self, acceleration):
    newSpacing = self.leader.location - self.follower.location
    newTimeGap = newSpacing / (self.follower.speed + .001)
    expected_speed = 33.5
    expected_time_gap = 1

    if self.rewardName == 'test':
      reward = 0
      # headway
      if newTimeGap < 0.6:
        reward -= expected_speed
      # speed reward
      reward += (expected_speed - max(0, expected_speed - self.follower.speed))
      if newSpacing < 0:
        reward -= 10000
      return reward

    elif self.rewardName == 'xiaoboqu':
      reward = 0
      # headway
      if newTimeGap < 1:
        reward -= expected_speed
      # speed reward
      reward += (expected_speed - max(0, expected_speed - self.follower.speed))
      if newSpacing < 0:
        reward -= 100
      return reward

    elif self.rewardName == 'cathaywu':
      expected_speed = 18
      speed_reward = expected_speed - max(0, expected_speed - self.follower.speed)
      reward = speed_reward
      if newSpacing < 0:
        reward -= 100
      return reward
    elif self.rewardName == 'original':
      # headway
      reward = 0
      if newSpacing < 0:
        reward -= HIT_PENALTY
      else:
        reward -= HIT_PENALTY * max(0, 1 - newTimeGap)
      # speed
      reward += max(expected_speed, self.follower.speed) * SPEED_REWARD
      reward += (1 - newTimeGap) * (self.leader.speed - self.follower.speed) * SPEED_DIFF_REWARD
      return reward

    elif self.rewardName == 'liming':
      alpha = 1
      beta = 1
      gamma = 1
      delta = 4
      reward = 0

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward

    elif self.rewardName == 'liming_no_acceleration_term':
      alpha = 1
      beta = 1
      gamma = 1
      delta = 0
      reward = 0

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward

    elif self.rewardName == 'liming_large_time_gap':
      alpha = 1
      beta = 1
      gamma = 1
      delta = 4
      reward = 0

      expected_time_gap = 2

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward

    elif self.rewardName == 'liming_strong_efficient':
      alpha = 1
      beta = 2
      gamma = 1
      delta = 4
      reward = 0

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward

    elif self.rewardName == 'liming_safer':
      alpha = 1.2
      beta = 1
      gamma = 1
      delta = 4
      reward = 0

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward

    elif self.rewardName == 'standard':
      alpha = self.SAFETY_WEIGHT
      beta = self.EFFICIENCY_WEIGHT
      gamma = self.COMFORT_WEIGHT
      delta = 4
      reward = 0

      # time headway
      if expected_time_gap > newTimeGap > 0:
        reward -= alpha * (100 - 100 * np.sqrt(max(0, expected_time_gap-(expected_time_gap-newTimeGap)**2)))

      # speed reward
      reward += beta * min(expected_speed, self.follower.speed)

      #speed diff reward
      if newTimeGap < expected_time_gap and self.leader.speed > self.follower.speed:
        reward += gamma * (self.leader.speed - self.follower.speed) * (expected_time_gap - newTimeGap)

      reward -= delta * acceleration ** 2

      return reward