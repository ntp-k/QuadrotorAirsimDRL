
# advanced reward function
'''
if self.state["collision"]: # collide
    rewards = -1000
    done = True
elif self.state["position"].z_val < self.MAX_ALTITUDE or self.state["position"].z_val > self.MIN_ALTITUDE or self.state["position"].y_val < self.MAX_LEFT or self.state["position"].y_val > self.MAX_RIGHT or self.state["position"].x_val < self.MAX_SOUTH or self.state["position"].x_val > self.MAX_NORTH: # out of range
    rewards = -100
    done = True
else: # grant reward
    # if self.middle_pixel > 246:
    #     reward_avoiding = 5

    distance = np.linalg.norm(self.destination - quad_pt)
    prev_distance = np.linalg.norm(self.destination - prev_quad_pt) 

    if distance >= prev_distance: # move opposite of destination or stay still
        rewards = 0
    else: # reach destination in range 10 meter
        if distance <= 10:
            reward_dist = 1000
            done = True
        else: # move toward destination, raward is % if travelled distance
            reward_dist = ((self.destination[0] - distance) / self.destination[0]) * 100

        reward_speed = np.linalg.norm([self.state["velocity"].x_val,
                                        self.state["velocity"].y_val,
                                        self.state["velocity"].z_val,])

    rewards = reward_dist + reward_speed + reward_avoiding
'''




'''v11_dqn_multiInput_4actions_2obs_simpleRF_100000_steps_1

proceed to destination + collision avoidance

train
policy : MultiInput
timestep : 100,000 (60,000)

env
obs : image, position
action : 4

map
goal : 300 meters
obstacle : 3 static, manual
'''

def _get_obs(self):
    responses = self.drone.simGetImages([self.image_request])
    image = self.transform_obs(responses)
    self.drone_state = self.drone.getMultirotorState()

    self.state["prev_position"] = self.state["position"]
    self.state["position"] = self.drone_state.kinematics_estimated.position
    self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
    quad_pt = np.array(list((self.state["position"].x_val,
                             self.state["position"].y_val,
                             self.state["position"].z_val,)))

    collision = self.drone.simGetCollisionInfo().has_collided
    self.state["collision"] = collision

    if collision:
        collision = 1
    else:
        collision = 0
    
    obs = dict()
    obs = {'depth_cam': image,
           'position': quad_pt}
        #    'collision': collision}

    return obs

def transform_obs(self, responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = np.full(img1d.size, 255, dtype=np.float ) - (255/img1d)
    # img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    

    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert("L"))
    # self.middle_pixel = im_final[41][41]

    return im_final.reshape([84, 84, 1])

def _compute_reward(self, action):
    rewards = float(0.0)
    # reward_dist = 0
    distance = 0.0
    done = False
    quad_pt = np.array(list((self.state["position"].x_val,
                                self.state["position"].y_val,
                                self.state["position"].z_val,)))


    # simple reward function
    distance = np.linalg.norm(self.destination - quad_pt)

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif distance <= 10: # reach destination within range of 10
        rewards = 100
        done = True
    else:
        rewards = self.distance - distance

    self.total_rewards += rewards
    if self.total_rewards <= -100:
        done = True
        
    
    self.distance = distance

    return rewards, done



'''v12_dqn_cnnPolicy_4actions_imageObs_10000_steps

proceed forward + collision avoidance

train
policy : cnn
timestep : 10000

env
obs : image
action : 4

map
goal : 200 meters
obstacle : 3 random, every 50 meters
'''

def _get_obs(self):
    responses = self.drone.simGetImages([self.image_request])
    image = self.transform_obs(responses)
    self.drone_state = self.drone.getMultirotorState()

    self.state["prev_position"] = self.state["position"]
    self.state["position"] = self.drone_state.kinematics_estimated.position
    self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

    collision = self.drone.simGetCollisionInfo().has_collided
    self.state["collision"] = collision


    return image

def _compute_reward(self, action):
    rewards = float(0.0)
    done = False
    quad_pt = np.array(list((self.state["position"].x_val,
                                self.state["position"].y_val,
                                self.state["position"].z_val,)))

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif self.state["position"].x_val > 200:
        rewards = 100
        done = True
    elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        done = True
    elif action == 0:
        rewards = 1
    
    return rewards, done


'''v13_dqn_cnnPolicy_4actions_imageObs_10000_steps

proceed forward + collision avoidance

change : position[x] > 200 no longer give rewards

train
policy : cnn
timestep : 10000

env
obs : image
action : 4

map
goal : 200 meters
obstacle : 3 random, every 50 meters
'''
def _compute_reward(self, action):
    rewards = float(0.0)
    done = False
    quad_pt = np.array(list((self.state["position"].x_val,
                                self.state["position"].y_val,
                                self.state["position"].z_val,)))

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif self.state["position"].x_val > 200:
        done = True
    elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        done = True
    elif action == 0:
        rewards = 1
    
    return rewards, done


'''v14_dqn_cnnPolicy_4actions_imageObs_10000_steps

proceed forward + collision avoidance

change : position[x] > 200 give rewards, action == 0 no longer give rewards

train
policy : cnn
timestep : 10000

env
obs : image
action : 4

map
goal : 200 meters
obstacle : 3 random, every 50 meters
'''
def _compute_reward(self, action):
    rewards = float(0.0)
    done = False
    quad_pt = np.array(list((self.state["position"].x_val,
                                self.state["position"].y_val,
                                self.state["position"].z_val,)))

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif self.state["position"].x_val > 200:
        done = True
        rewards = 100
    elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        done = True
    
    return rewards, done


# failed at 800 timestep
'''v14_dqn_cnnPolicy_4actions_imageObs_10000_steps

proceed forward + collision avoidance

change :
1. depth image cut miximum at 255 (not scale, but round to 255)
2. get rewards when avoid building and minus when collide
3. env guranntee with block rate 

train
policy : cnn
timestep : 10000

env
obs : image, ( 0-255 ; any value exceed 255 will be rounded down to 255)
action : 4

map
goal : 200 meters
obstacle :  random, every 50 meters (block rate 70%)
'''

def _compute_reward(self, action):
    rewards = float(0.0)
    done = False
    pass1 = pass2 = pass3 = False

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        done = True
    elif self.state["position"].x_val > 200:
        rewards = 25
        done = True
    elif self.state["position"].x_val > 150 and not pass3:
        rewards = 25
        pass3 = True
    elif self.state["position"].x_val > 100 and not pass2:
        rewards = 25
        pass2 = True
    elif self.state["position"].x_val > 50 and not pass1:
        rewards = 25
        pass1 = True

    return rewards, done

# train v14 again
'''v15_dqn_cnnPolicy_4actions_imageObs_10000_steps
'''

'''v16_dqn_cnnPolicy_4actions_imageObs_10000_steps

proceed forward + collision avoidance

change : pass1, pass2, pass3 are now self.pass123

train
policy : cnn
timestep : 10000

env
obs : image, ( 0-255 ; any value exceed 255 will be rounded down to 255)
action : 4

map
goal : 200 meters
obstacle :  random, every 50 meters (block rate 70%)

traning result has a good trend.
'''
def _compute_reward(self, action):
    rewards = float(0.0)
    done = False

    if self.state["collision"]: # collide
        rewards = -100
        done = True
    elif self.state["position"].x_val < -80 or self.state["position"].y_val > 80 or self.state["position"].y_val < -80 :
        done = True
    elif self.state["position"].x_val > 200:
        rewards = 25
        done = True
    elif self.state["position"].x_val > 150 and not self.pass3:
        rewards = 25
        self.pass3 = True
    elif self.state["position"].x_val > 100 and not self.pass2:
        rewards = 25
        self.pass2 = True
    elif self.state["position"].x_val > 50 and not self.pass1:
        rewards = 25
        self.pass1 = True

    return rewards, done