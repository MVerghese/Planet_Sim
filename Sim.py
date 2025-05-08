import numpy as np
from matplotlib import pyplot as plt
import json
import os
from tqdm import tqdm

np.set_printoptions(suppress=True)

def base_control_policy(objects, masses, positions, velocities):
    """
    Base control policy to update positions and velocities.
    """
    return np.zeros(3) # Placeholder for control policy

def gravity(m1, m2, r):
    """
    Calculate the gravitational force between two masses.
    """
    G = 6.67430e-11  # gravitational constant
    return G * m1 * m2 / r**2

def gravitational_forces(positions, masses):
    """
    Update accelerations based on current positions and masses.
    """
    n = len(positions)
    forces = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i != j:
                vector = np.array(positions[j]) - np.array(positions[i])
                distance = np.linalg.norm(vector)
                force = gravity(masses[i], masses[j], distance)
                forces[i] += force * (vector / distance)
    return forces

def load_initial_conditions(file_path):
        """
        Load initial conditions from a JSON file.
        """
        with open(file_path, 'r') as file:
            initial_conditions = json.load(file)
        objects = [key for key in initial_conditions.keys()]
        masses = [initial_conditions[key]['mass'] for key in initial_conditions.keys()]
        positions = [initial_conditions[key]['position'] for key in initial_conditions.keys()]
        velocities = [initial_conditions[key]['velocity'] for key in initial_conditions.keys()]
        return objects, masses, positions, velocities


class Simulation:
    def __init__(self, initial_conditions_file, control_policies):
        """
        Initialize the simulation with initial conditions and control policy.
        """
        self.objects, self.masses, self.positions, self.velocities = load_initial_conditions(initial_conditions_file)
        self.control_policies = [policy if callable(policy) else base_control_policy for policy in control_policies]

    def setup_dynamics(self):
        n_objects = len(self.objects)
        A = np.block([
            [np.zeros((3,3)), np.eye(3)],
            [np.zeros((3,3)), np.zeros((3,3))]
        ])
        self.A_block = np.zeros((6*n_objects, 6*n_objects))
        for i in range(n_objects):
            self.A_block[i*6:(i+1)*6, i*6:(i+1)*6] = A
        B_stack = [np.vstack((np.zeros((3,3)),np.eye(3)*1/m)) for m in self.masses]
        self.B_block = np.zeros((6*n_objects, 3*n_objects))
        for i in range(n_objects):
            self.B_block[i*6:(i+1)*6, i*3:(i+1)*3] = B_stack[i]
        
        self.C_block = self.B_block.copy()
    
    def dynamics(self,x_s,u_s,d_s):
        """
        Calculate x_dot where x = [positions, velocities].
        Follows x_dot = Ax + Bu + Cd
        where A is the state matrix, B is the control matrix, and C is the acceleration matrix.
        """
        x_dot = self.A_block @ x_s + self.B_block @ u_s + self.C_block @ d_s
        return x_dot
    
    def update_dynamics(self, h):
        """
        Update the dynamics of the simulation.
        """
        n_objects = len(self.objects)
        x_s = [np.concatenate((self.positions[i], self.velocities[i])) for i in range(n_objects)]
        x_s = np.concatenate(x_s)
        
        u_s = [self.control_policies[i](self.objects, self.masses, self.positions, self.velocities) for i in range(n_objects)]
        u_s = np.concatenate(u_s)
        
        d_s = gravitational_forces(self.positions, self.masses)
        d_s = np.concatenate(d_s)
        
        # Runge-Kutta 4th order method
        k1 = self.dynamics(x_s, u_s, d_s)
        # import pdb; pdb.set_trace()
        k2 = self.dynamics(x_s + h*k1/2, u_s, d_s)
        k3 = self.dynamics(x_s + h*k2/2, u_s, d_s)
        k4 = self.dynamics(x_s + h*k3, u_s, d_s)
        x_s_new = x_s + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x_s_new = np.reshape(x_s_new, (n_objects, 6))
        self.positions = [x_s_new[i][:3] for i in range(n_objects)]
        self.velocities = [x_s_new[i][3:] for i in range(n_objects)]
        
        return x_s_new

    def run_simulation(self, time_step, total_time):
        """
        Run the simulation for a given time step and total time.
        """
        self.setup_dynamics()
        num_steps = int(total_time / time_step)
        x_s_array = np.zeros((num_steps, len(self.objects), 6))
        for step in tqdm(range(num_steps)):
            x_s_new = self.update_dynamics(time_step)
            x_s_array[step] = x_s_new
        return x_s_array

    def plot_simulation_data(self,simulation_data,save_folder, vis_scale = 1):
        vis_config_path = 'vis_config.json'
        with open(vis_config_path, 'r') as file:
            vis_config = json.load(file)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        max_pos = np.max(np.abs(simulation_data[:, :, :3]))*1.1
        fig = plt.figure(figsize=(10, 10))
        for step in tqdm(range(simulation_data.shape[0])):
            plt.clf()
            ax = plt.gca()
            for i, obj in enumerate(self.objects):
                pos = simulation_data[step, i, :3]
                circle = plt.Circle((pos[0], pos[1]), vis_config[obj]["radius"]*vis_scale, color=vis_config[obj]["color"])
                ax.add_patch(circle)
            ax.set_xlim(-max_pos, max_pos)
            ax.set_ylim(-max_pos, max_pos)
            ax.set_aspect('equal')
            plt.title(f"Step: {step}")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.savefig(os.path.join(save_folder, "step_"+str(step).zfill(6)+".jpg"))




def main():
    initial_conditions_file = 'initial_conditions.json'
    # initial_conditions_file = 'initial_conditions_test.json'
    control_policies = [base_control_policy] * 2  # Example: same policy for all objects
    sim = Simulation(initial_conditions_file, control_policies)
    
    # Example: simulate the moon for a month
    time_step = 1.0  # seconds
    total_time = 1.0*3600*24*30  # seconds
    print("Running simulation...")
    simulation_data = sim.run_simulation(time_step, total_time)
    simulation_data = simulation_data[::3000, :, :]  # Downsample for visualization
    # for i in range(simulation_data.shape[0]):
    #     print(f"Step {i}: \n{simulation_data[i]}")
    save_folder = 'simulation_results'
    print("Generating visualization...")
    sim.plot_simulation_data(simulation_data, save_folder, vis_scale=1)
    print(f"Simulation data saved to {save_folder}")
    
if __name__ == "__main__":
    main()