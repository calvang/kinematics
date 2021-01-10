import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Arm:
    """
    A 6-axis robotic arm that is composed of 2D-constrined links with 
    a turntable base centered at (x,y,z) = (0,0,0). 
    """
    def __init__(self, base_angle, links, angles, angle_constraints, radians=False):
        """Initialize arm and create numpy arrays"""
        self.num_joints = len(links) # number of joints
        self.positions = np.zeros((self.num_joints ,2))
        self.links = np.array(links) # length of preceding link
        if not radians:
            self.angles = np.radians(np.array(angles))
            self.base_angle = np.radians(base_angle) # base angle relative to world
        else:
            self.angles = np.array(angles)
            self.base_angle = base_angle # base angle relative to world
        self.length = np.sum(self.links)
        self.angle_constraints = np.array(angle_constraints)
        self.base_position = np.array((0,0)) # TODO change this
        self.base_world_position = np.array((0,0,0))
        self.forward_kinematics()

    def check_status(self):
        if num_joints == 0: return False
        else: return True

    def forward_kinematics(self):
        """Given a set of angles, calculate the coordinates of the hand"""
        prev_angle = self.angles[0]
        prev_joint = self.positions[0].copy()
        for i in range(1, self.num_joints):
            prev_joint += np.array((self.links[i]*np.cos(prev_angle), 
                                    self.links[i]*np.sin(prev_angle)))
            prev_angle += self.angles[i]
            self.positions[i] = prev_joint.copy()
        # print(prev_joint)
        return prev_joint

    def distance_from_target(self, target):
        current = self.forward_kinematics()
        return distance.euclidean(current, target)

    def inverse_kinematics_sgd(self, target):
        """
        Perform stochastic gradient descent to approximate inverse kinematics
        using a 2D coordinate as the target

        Params:
        - target: (2,0) numpy array containing the x,y coordinates of the target
        """
        stochastic_dist = 0.00001
        lr = 0.001
        eps = 0.01 # distance theshold
        if distance.euclidean(self.base_position, target) > self.length:
            print("Target out of range:", target)
            exit(1)
        while self.distance_from_target(target) > eps:
            for i in range(0, self.num_joints):
                orig_angle = self.angles[i].copy()
                fx = self.distance_from_target(target)
                self.angles[i] += stochastic_dist
                fx_d = self.distance_from_target(target)
                s_grad = (fx_d - fx) / stochastic_dist
                self.angles[i] = orig_angle - lr * s_grad

    def jacobian_pseudoinverse(self):
        """Calculate a 2D Jacobian Pseudoinverse"""
        j_t = np.empty((self.num_joints-1, 2)) # transpose dims
        prev_angle = self.angles[0]
        prev_partial = np.zeros(2)
        for i in range(1, self.num_joints):
            partial = np.array( # partial derivative in column of jacobian
                [-1*self.links[i]*np.sin(prev_angle),
                self.links[i]*np.cos(prev_angle)])
            prev_angle += self.angles[i]
            j_t[i-1] = prev_partial + partial
            prev_partial = j_t[i-1]
        return j_t

    def jacobian_transpose(self):
        """Calculate the 2D Jacobian Transpose"""
        j_t = np.zeros((self.num_joints-1, 2)) # transpose dims
        prev_angle = self.angles[0]
        for i in range(1, self.num_joints):
            partial = np.array( # partial derivative in column of jacobian
                [-1*self.links[i]*np.sin(prev_angle),
                self.links[i]*np.cos(prev_angle)])
            prev_angle += self.angles[i]
            j_t[:i-1] += + partial
        return j_t

    def inverse_kinematics_jt(self, target):
        """
        Perform Jacobian Transpose to calculate inverse kinematics
        using a 2D coordinate as the target
        """
        lr = 0.01
        eps = 0.01 # distance theshold
        if distance.euclidean(self.base_position, target) > self.length:
            print("Target out of range:", target)
            exit(1)
        current_e = self.forward_kinematics()
        curr_dist = distance.euclidean(current_e, target)
        while curr_dist > eps:# and np.abs(curr_dist - prev_dist) > 0.00001:
            j_t = self.jacobian_transpose()
            v = np.abs(target - current_e) # spatial diff
            d0 = (j_t @ v) / np.sin(self.angles[:-1]) # change in orientation
            self.angles[:-1] += lr * d0
            current_e = self.forward_kinematics()
            curr_dist = distance.euclidean(current_e, target)

    def inverse_kinematics_jpi(self, target):
        """
        Perform Jacobian Transpose to calculate inverse kinematics
        using a 2D coordinate as the target

        Params:
        - target: (2,0) numpy array containing the x,y coordinates of the target
        """
        lr = 0.01
        eps = 0.01 # distance theshold
        if distance.euclidean(self.base_position, target) > self.length:
            print("Target out of range:", target)
            exit(1)
        current_e = self.forward_kinematics()
        curr_dist = distance.euclidean(current_e, target)
        while curr_dist > eps:# and np.abs(curr_dist - prev_dist) > 0.00001:
            j_pi = self.jacobian_pseudoinverse()
            v = np.abs(target - current_e) # spatial diff
            d0 = (j_pi @ v) / np.sin(self.angles[:-1]) # change in orientation
            self.angles[:-1] += lr * d0
            current_e = self.forward_kinematics()
            curr_dist = distance.euclidean(current_e, target)

    def signed_arctan(self, coord):
        # print("angle", np.arctan(coord[1] / coord[0])*180/np.pi)
        # print("offset", np.pi * (1 - np.sign(coord[0])) / 2)
        return np.arctan(coord[1] / coord[0]) + np.pi * (1 - np.sign(coord[0])) / 2

    def reach(self, head, tail, link):
        r = distance.euclidean(self.positions[head], self.positions[tail])
        scale = link / r
        self.positions[tail] = (1 - scale) * self.positions[head] \
                               + scale * self.positions[tail]

    def update_angles(self):
        """Calculate arm angles after FABRIK"""
        prev_angle = 0
        for i in range(1, self.num_joints):
            curr_vec = self.positions[i] - self.positions[i-1]
            curr_angle = self.signed_arctan(curr_vec) - prev_angle
            if curr_angle > np.pi: curr_angle -= 2*np.pi
            elif curr_angle < -np.pi: curr_angle += 2*np.pi
            self.angles[i-1] = curr_angle
            prev_angle += curr_angle

    def inverse_kinematics_fabrik(self, target):
        """
        Perform forward and backward reaching inverse kinematics solver
        using a 2D coordinate as the target

        Params:
        - target: (2,0) numpy array containing the x,y coordinates of the target
        """
        eps = 0.01 # distance theshold
        if distance.euclidean(self.base_position, target) > self.length:
            print("Target out of range:", target)
            exit(1)
        current_e = self.forward_kinematics()
        curr_dist = distance.euclidean(current_e, target)
        while curr_dist > eps: # usually only takes 1 iteration
            self.positions[-1] = target
            # forward reaching
            for i in range(self.num_joints-2, 0, -1): # start at end
                self.reach(i+1, i, self.links[i+1])
            # print(self.positions)
            # backward reaching
            for i in range(1, self.num_joints-1):
                self.reach(i, i+1, self.links[i+1])
            current_e = self.positions[-1]
            curr_dist = distance.euclidean(current_e, target)
        self.update_angles() # update angles based on joint positions

    def inverse_kinematics_base(self, target): 
        """
        Calculate the angle of the base and adjust accordingly
        using a 3D coordinate as the target

        Params:
        - target: (3,) numpy array of x,y,z coordinates
        Returns:
        - (2,) numpy array: projection of the target onto the plane 
                            given by the base angle
        """
        xz_comp = np.array([target[0], target[2]]) # (x,z) components
        new_base_angle = self.signed_arctan(xz_comp)
        if new_base_angle > np.pi: new_base_angle -= 2*np.pi
        elif new_base_angle < -np.pi: new_base_angle += 2*np.pi
        # print("New angle", new_base_angle * 180 / np.pi)
        # print("Angle change", (new_base_angle - self.base_angle) * 180 / np.pi)
        self.base_angle = new_base_angle
        h = distance.euclidean(self.base_position, xz_comp)
        return np.array((h, target[1]))   
        
    def move_to(self, target, ik="fabrik"):
        """
        Move the arm's end effector to the 3D target using inverse kinematics

        Params:
        - target: a length 3 list or tuple containing x,y,z coordinates
        - ik: the inverse kinematics algorithm to use, defaults to FABRIK
        """
        target_proj = self.inverse_kinematics_base(np.array(target)) # 2D projection of target
        if ik == "fabrik": # 2D FABRIK Solver
            self.inverse_kinematics_fabrik(target_proj)
        elif ik == "sgd": # 2D SGD Solver
            self.inverse_kinematics_sgd(target_proj)
        elif ik == "jt": # 2D Jacobian Transpose Solver
            self.inverse_kinematics_jt(target_proj)

    def change_angle(self, joint, delta):
        """Move a joint's angle by a given delta and return success state"""
        new_angle = self.angles[joint] + delta
        if self.angle_constraints[joint][0] <= new_angle and \
            self.angle_constraints[joint][1] >= new_angle:
            self.angles[joint] = new_angle
            self.forward_kinematics() # update positions
            return True
        else:
            return False

    def plot(self, file):
        """Plot a 2D representation of the join configuration"""
        plt.figure()
        plt.plot(self.positions[:,0], self.positions[:,1])
        plt.xlim(-10, 10)
        plt.ylim(0, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(file)
