#!/usr/bin/env python
import rospy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

from ambf_msgs.msg import ObjectState, ObjectCmd
from kuka7DOF_Spatial_Transform_case import *

import numpy as np

# KukaController class implementation
# for hybrid pd impedance controller only
class KukaController:

	# Initialize node, subscriber and publisher
	def __init__(self):
		rospy.init_node('kuka_gravity_compensation')
		self.sub = rospy.Subscriber('/ambf/env/base/State', ObjectState, self.getJointState, queue_size=1)
		self.pub = rospy.Publisher('/ambf/env/base/Command', ObjectCmd, queue_size=1)
		self.initialize_controller()

	# Initialize controller
	def initialize_controller(self):
		cmd_msg = ObjectCmd()
		cmd_msg.enable_position_controller = False
		cmd_msg.position_controller_mask = [False]

		self.NJoints = get_joint_num()
		self.cmd_msg = cmd_msg
		self.t0 = rospy.get_time()
		self.prev_time = 0
		self.prev_time_PD = 0

		## Joint Space
		self.prev_state_PD = np.zeros(5) #the last 5 joints are controller using PID

		self.prev_state = np.zeros(3)
		self.XGoal = np.zeros(3)

		self.prev_J = 0*np.eye(3)

		## Task Space
		self.prev_x = np.zeros(3)
		self.prev_velX = np.zeros(3)

		# Set end-effector desired velocity here
		self.X_SPEED = np.zeros(3)
		self.X_SPEED[1] = .5

		self.x_speed_local = self.X_SPEED

		# Set desired joint velocity here
		self.Q_SPEED = np.zeros(5)

		self.joint_speed = self.Q_SPEED

		self.switch = 0

	def getJointState(self, data):
		self.state = np.asarray(data.joint_positions)
		self.t = rospy.get_time() - self.t0 #clock
		self.Hybrid_PD_Impedance()


	def generate_joint_trajectory(self, q):
		print "case: ",self.switch
		if self.switch == 0:
			q2ref = 0
			print q[0]
			self.joint_speed = self.Q_SPEED
			if self.t > 10:
				self.switch = 1
		elif self.switch == 1:
			q2ulim = np.pi/9
			print self.state[0]
			self.joint_speed = self.Q_SPEED
			self.joint_speed[0] = 1
			if q[0] > q2ulim:
				self.switch = 2
		elif self.switch == 2:
			q3llim = np.pi/3
			print q[1]
			self.joint_speed = self.Q_SPEED
			self.joint_speed[1] = -1
			if q[1] < q3llim:
				self.switch = 3
		elif self.switch == 3:
			q2llim = -np.pi/9
			print q[0]
			self.joint_speed = self.Q_SPEED
			self.joint_speed[0] = -1
			if q[0] < q2llim:
				self.switch = 4
		elif self.switch == 4:
			q3ulim = np.pi/2
			print q[1]
			self.joint_speed = self.Q_SPEED
			self.joint_speed[1] = 1
			if q[1] > q3ulim:
				self.switch = 1

		q_des = q + self.joint_speed * (self.dt)
		velQ_des = self.joint_speed
		return [q_des, velQ_des]

	def Hybrid_PD_Impedance(self):

		# ============================== Impedance controller ==============================

		# Desired stiffness
		Kd = np.diag([0.1, 0.01, 0.1])

		# Desired damping
		Dd = np.diag([0.0001, 0.0001, 0.0001])

		# Desired inertia
		Md = np.diag([0.02, 0.02, 0.02])

		# Time step
		self.dt = self.t - self.prev_time

		# Get position for the end of 4th link (including base link)
		x = get_end_effector_pos_link3(self.state)

		# Constant position reference for the impedance controller (no trajectory generation)
		self.XGoal[0] = -0.0
		self.XGoal[1]= 0.05
		self.XGoal[2] = 0.5

		# Velocity and acceleration reference
		XvelGoal = np.zeros(3)
		XaccGoal = np.zeros(3)

		# Impedance controller joint velocity
		vel = (self.state[:3] - self.prev_state)/self.dt

		# Task space velocity
		velX = (x - self.prev_x)/self.dt

		# Task space position and velocity error
		errX = np.asarray(self.XGoal - x)
		derrX = XvelGoal - velX

		# Geometric jacobian for first 3 joints (3x3)
		J = get_end_effector_jacobian_link3(self.state)
		J1 = J[:,:3]

		# Jacobian inverse
		J_inv = np.linalg.pinv(J1)

		# Joint space inertia matrix
		Mq = get_M(self.state)
		Mq1 = Mq [:3,:3]

		# Task space inertia matrix
		Mx = np.dot(np.dot(np.transpose(J_inv),Mq1),J_inv)

		# Impedance controller force
		F = np.dot(np.dot(np.linalg.inv(Md), Mx),( np.dot(Dd, derrX) + np.dot(Kd, errX) ) )

		# Joint space gravity term
		G = get_G(self.state)
		G3 = G[:3]
		G3[2] = 0 # set gravity term of the 3rd joint to 0, it will be included in PD controller

		# Torque input from impedance controller
		tau_ = G3 + np.dot(np.transpose(J1),F)

		# Store previous values
		self.prev_time = self.t
		self.prev_state = self.state[:3]
		self.prev_x = x

		# ============================== PD Controller ==============================

		NJoints_PD = 5 # tip, ball, two revolutes

		# Initializing the PD gain values
		K_P = np.diag([1,1,0.01,0.01,0.01])
		K_D = np.diag([0.07,0.07,0.0001,0.0001,0.0001])

		# PD controller joint position vector
		q_PD = self.state[2:]

		# Generate joint space trajectory
		q_traj = self.generate_joint_trajectory(q_PD)
		stateGoal_PD = q_traj[0]

		# Switch cases for square edges
		if self.switch==0: # initialization
			stateGoal_PD[0] = 0
			stateGoal_PD[1] = np.pi/2
			stateGoal_PD[3] = -stateGoal_PD[0]

		elif self.switch==1: # bottom edge
			stateGoal_PD[1] = np.pi/2
			stateGoal_PD[3] = -stateGoal_PD[0]

		elif self.switch==2: # left edge
			stateGoal_PD[0] = np.pi/9
			stateGoal_PD[3] = -self.state[1] + stateGoal_PD[1] - np.pi/2

		elif self.switch==3: # top edge
			stateGoal_PD[1] = np.pi/3
			stateGoal_PD[3] = stateGoal_PD[0]

		elif self.switch==4: # right edge
			stateGoal_PD[0] = -np.pi/9
			stateGoal_PD[3] = -self.state[1] + stateGoal_PD[1] - np.pi/2

		# constant references
		stateGoal_PD[2] = np.pi/2
		stateGoal_PD[4] = 0

		# velocity reference:
		velGoal_PD = q_traj[1]

		# time computations:
		time = rospy.get_time()
		dt = time - self.prev_time_PD
		self.prev_time_PD = time

		# PD controller joint velocity
		vel_PD = (self.state[2:] - self.prev_state_PD)/dt

		# Joint space gravity term
		G = get_G(self.state)
		G2 = G[2:]

		# Joint space position and velocity error
		err_PD = np.asarray(stateGoal_PD - self.state[2:])
		derr_PD = velGoal_PD - vel_PD

		# PD controller torque input
		tau = G2 + np.dot(K_P,err_PD) + np.dot(K_D,derr_PD)

		# Generate torque input message
		self.cmd_msg.joint_cmds = [tau_[0], tau_[1], tau_[2]+tau[0],tau[1], tau[2], tau[3], tau[4]]

		# Publish torque input
		self.pub.publish(self.cmd_msg)

		# Store previous values
		self.prev_state_PD = self.state[2:]
		self.prev_vel_PD = vel_PD
		self.prev_x = x
		self.prev_J = J1

if __name__ == '__main__':
	start = KukaController()
	rospy.spin()
