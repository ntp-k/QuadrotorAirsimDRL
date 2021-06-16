

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