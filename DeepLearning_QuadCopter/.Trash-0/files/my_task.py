import numpy as np
from physics_sim import PhysicsSim

class Take_Off():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_z=None):
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
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_z = target_z if target_z is not None else 10

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 0
        penalty = 0
        
        # penalty for euler angles, we want the takeoff to be stable
        penalty += abs(self.sim.pose[3:6]).sum()
        # penalty for movement in x/y dimension, we want it to go up as straight as possible
        # since we are more concerned about shifting left/right or forward/backward in xy plane, we multiple it by 2
        penalty += 2*(abs(self.sim.pose[0:2]).sum())
        
        #Since reaching a desired height is the goal, failing on that has the highest penalty
        # But it also gets reward if the achedived height is closer to the target
        distance_to_target = abs(self.sim.pose[2]-self.target_z)
        
        penalty += distance_to_target**2

        
        if distance_to_target < 10: # this implies that agent is close to the target
            reward += 1000

        reward += 100
#         print("\rMy task get_reward: distance_to_target = {}, reward={}, penalty={}, final_reward={},my current_pos={},target_z={}".format(
#                 distance_to_target, reward, penalty,reward - penalty*0.003, self.sim.pose, self.target_z), end="")  # [debug]
        return reward - penalty*0.003
        
    

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state