from ambf_client import Client
import time
from kuka7DOF_Spatial_Transform_case import *
import numpy as np
from scipy.spatial.transform import Rotation as R

c = Client()

c.connect()

base = c.get_obj_handle('base')

time.sleep(0.1)

base.set_joint_pos(0,0)
time.sleep(0.2)
base.set_joint_pos(1,1.5708)
time.sleep(0.2)
base.set_joint_pos(2,0)
time.sleep(0.2)
base.set_joint_pos(3,1.5708)
time.sleep(0.2)
base.set_joint_pos(4,-1.5708)
time.sleep(0.2)
base.set_joint_pos(5,1.5708)
time.sleep(0.2)
base.set_joint_pos(6,0)
time.sleep(3)
q = np.zeros(6)
for i in range(0,6):
    q[i] = base.get_joint_pos(i)

print q
x_pos = get_end_effector_pos(q)
print x_pos
print np.linalg.inv(get_rot(q))
orientation = R.from_dcm(np.linalg.inv(get_rot(q)))
print orientation
x_orientation = orientation.as_euler('zyx', degrees=True)
print x_orientation
J = get_6_jacobian(q)
print "J(1-3) = ", J[0:3]
print "J(4-6) = ", J[3:]
B = np.array([[1,0,np.sin(x_orientation[1])],[0,np.cos(x_orientation[0]),-np.cos(x_orientation[1])*np.sin(x_orientation[0])],[0,np.sin(x_orientation[0]),np.cos(x_orientation[1])*np.cos(x_orientation[0])]])
print "B = ", B
transform_B = np.zeros((6,6))
transform_B[0:3,0:3] = np.eye(3)
transform_B[3:,3:] = B
print transform_B
time.sleep(40)
