import numpy as np
from physics_sim import PhysicsSim

class Task_TakeOff():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, runtime=5., target_z=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_z: target/goal (z) position for the agent
        """
        # Simulation
        init_pose = np.array([0., 0., np.random.normal(0.5, 0.1), 0., 0., 0.])  # initial pose at ground but  
        # drop off from a slight random height otherwise thurst is not enough to takeoff
        init_velocities = np.array([0., 0., 0.])         # initial velocities
        init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities

        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.state_range = self.sim.upper_bounds[2] - self.sim.lower_bounds[2]
        
#         print(self.state_range)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_z = 15
        # minimum target height (z position) to reach for successful takeoff
        self.counter=1

       
    
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_z)).sum()
        reward = -min(abs(self.target_z - self.sim.pose[2]), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
#         print('task get_reward reward')
#         print(reward)
        return reward
    
    
     
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        # Compute reward / penalty and check if this episode is complete
        done = False
        reward = -min(abs(self.target_z - self.sim.pose[2]), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            pose_all.append(self.sim.pose)
            if self.sim.pose[2] >= self.target_z:  # agent has crossed the target height
                reward += 10.0  # bonus reward
                done = True
            
        next_state = np.concatenate(pose_all)
#         print('task step next_state')
#         print(next_state)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
#         print('task reset [self.sim.pose]  self.action_repeat state')
#         print([self.sim.pose])
#         print(self.action_repeat)
#         print(state)
        return state