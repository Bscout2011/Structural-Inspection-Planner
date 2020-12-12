#!/usr/bin/env python2

from __future__ import print_function

import sys
import rospy
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.srv import GetPositionIKRequest
from moveit_msgs.srv import GetPositionIKResponse
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header
import numpy as np
from tf.transformations import *


class GetFK(object):
    def __init__(self, fk_link, frame_id):
        """
        A class to do FK calls thru the MoveIt!'s /compute_fk service.
        :param str fk_link: link to compute the forward kinematics
        :param str frame_id: frame_id to compute the forward kinematics
        into account collisions
        """
        rospy.loginfo("Initalizing GetFK...")
        self.fk_link = fk_link
        self.frame_id = frame_id
        rospy.loginfo("Asking forward kinematics for link: " + self.fk_link)
        rospy.loginfo("PoseStamped answers will be on frame: " + self.frame_id)
        self.fk_srv = rospy.ServiceProxy('bvr_SIM/compute_fk',
                                         GetPositionFK)
        rospy.loginfo("Waiting for /compute_fk service...")
        self.fk_srv.wait_for_service()
        rospy.loginfo("Connected!")
        self.last_js = None
        self.js_sub = rospy.Subscriber('bvr_SIM/joint_states',
                                       JointState,
                                       self.js_cb,
                                       queue_size=1)

    def js_cb(self, data):
        self.last_js = data

    def get_current_fk_pose(self):
        resp = self.get_current_fk()
        if len(resp.pose_stamped) >= 1:
            return resp.pose_stamped[0]
        return None

    def get_current_fk(self):
        while not rospy.is_shutdown() and self.last_js is None:
            rospy.logwarn("Waiting for a /joint_states message...")
            rospy.sleep(0.1)
        return self.get_fk(self.last_js)

    def get_fk(self, joint_state, fk_link=None, frame_id=None):
        """
        Do an FK call to with.
        :param sensor_msgs/JointState joint_state: JointState message
            containing the full state of the robot.
        :param str or None fk_link: link to compute the forward kinematics for.
        """
        if fk_link is None:
            fk_link = self.fk_link

        req = GetPositionFKRequest()
        req.header.frame_id = 'base_link'
        req.fk_link_names = [self.fk_link]
        req.robot_state.joint_state = joint_state
        try:
            resp = self.fk_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionFKResponse()
            resp.error_code = 99999  # Failure
            return resp




def initPoseStamped():
    ps = PoseStamped()
    ps.pose.position.x = 0.0
    ps.pose.position.y = 0.0
    ps.pose.position.z = 0.0
    return ps


def pose_from_vector3D(waypoint):
    #http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
    offset  = np.array([1.5, 2.5, 0], dtype=np.float64)

    pose= Pose()
    pose.position.x = waypoint[0] + offset[0]
    pose.position.y = waypoint[1] + offset[1]
    pose.position.z = waypoint[2] + offset[2]
    #calculating the half-way vector.
    u = [1,0,0]
    norm = np.linalg.norm(waypoint[3:])
    v = np.asarray(waypoint[3:])/norm 
    if (np.array_equal(u, v)):
        pose.orientation.w = 1
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
    elif (np.array_equal(u, np.negative(v))):
        pose.orientation.w = 0
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 1
    else:
        half = [u[0]+v[0], u[1]+v[1], u[2]+v[2]]
        pose.orientation.w = np.dot(u, half)
        temp = np.cross(u, half)
        pose.orientation.x = temp[0]
        pose.orientation.y = temp[1]
        pose.orientation.z = temp[2]
    norm = np.sqrt(pose.orientation.x*pose.orientation.x + pose.orientation.y*pose.orientation.y + 
        pose.orientation.z*pose.orientation.z + pose.orientation.w*pose.orientation.w)
    if norm == 0:
        norm = 1
    pose.orientation.x /= norm
    pose.orientation.y /= norm
    pose.orientation.z /= norm
    pose.orientation.w /= norm
    return pose

def getZRotation(vec):
    return np.arccos(vec[0]/np.linalg.norm(vec))

def getPoseFromBaseWaypoint(waypoint):
    base_position_vec = waypoint[0:3]
    base_orientation_vec = waypoint[3:]
    z_theta = getZRotation(base_orientation_vec)
    q = random_quaternion(np.random.random(3))
    q[0] = 0.0  #x
    q[1] = 0.0  #y
    q[2] = np.sin(z_theta/2)  #z
    q[3] = np.cos(z_theta/2)  #w 

    base_pos = PoseStamped()
    base_pos.header.frame_id = "map"
    base_pos.pose.position.x = base_position_vec[0]
    base_pos.pose.position.y = base_position_vec[1]
    base_pos.pose.position.z = base_position_vec[2]
    base_pos.pose.orientation.x = q[0]
    base_pos.pose.orientation.y = q[1]
    base_pos.pose.orientation.z = q[2]
    base_pos.pose.orientation.w = q[3]
    return base_pos

class GetIK(object):
    def __init__(self, group="main_arm_SIM", ik_timeout=1.0, ik_attempts=0,
                 avoid_collisions=False):
        """
        A class to do IK calls thru the MoveIt!'s /compute_ik service.
        :param str group: MoveIt! group name
        :param float ik_timeout: default timeout for IK
        :param int ik_attempts: default number of attempts
        :param bool avoid_collisions: if to ask for IKs that take
        into account collisions
        """
        print("Initalizing GetIK...")
        self.group_name = group
        self.ik_timeout = ik_timeout
        self.ik_attempts = ik_attempts
        self.avoid_collisions = avoid_collisions
        # print("Computing IKs for group: ", self.group_name)
        # print("With IK timeout: ", str(self.ik_timeout))
        # print("And IK attempts: ", str(self.ik_attempts))
        # print("Setting avoid collisions to: " ,
                    #   str(self.avoid_collisions))
        self.ik_srv = rospy.ServiceProxy('bvr_SIM/compute_ik',GetPositionIK)
        print("Waiting for /compute_ik service...")
        self.ik_srv.wait_for_service()
        print("Connected!")

    def get_ik(self, pose_stamped,
               base_pose_stamped = initPoseStamped(),
               group=None,
               ik_timeout=None,
               ik_attempts=None,
               avoid_collisions=None):
        """
        Do an IK call to pose_stamped pose.
        :param geometry_msgs/PoseStamped pose_stamped: The 3D pose
            (with header.frame_id)
            to which compute the IK.
        :param str group: The MoveIt! group.
        :param float ik_timeout: The timeout for the IK call.
        :param int ik_attemps: The maximum # of attemps for the IK.
        :param bool avoid_collisions: If to compute collision aware IK.
        """
        if group is None:
            group = self.group_name
        if ik_timeout is None:
            ik_timeout = self.ik_timeout
        if ik_attempts is None:
            ik_attempts = self.ik_attempts
        if avoid_collisions is None:
            avoid_collisions = self.avoid_collisions

        goal_pose_stamped = PoseStamped()
        goal_pose_stamped.pose.position.x = pose_stamped.pose.position.x - base_pose_stamped.pose.position.x
        goal_pose_stamped.pose.position.y = pose_stamped.pose.position.y - base_pose_stamped.pose.position.y
        goal_pose_stamped.pose.position.z = pose_stamped.pose.position.z - base_pose_stamped.pose.position.z
        goal_pose_stamped.pose.orientation = pose_stamped.pose.orientation
        goal_pose_stamped.header = pose_stamped.header
        # print("goal pose = ",goal_pose_stamped.pose.position)
        req = GetPositionIKRequest()
        req.ik_request.group_name = group
        req.ik_request.pose_stamped = goal_pose_stamped
        req.ik_request.timeout = rospy.Duration(ik_timeout)
        # req.ik_request.attempts = ik_attempts
        req.ik_request.avoid_collisions = avoid_collisions

        try:
            resp = self.ik_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionIKResponse()
            resp.error_code = 99999  # Failure
            return resp
def main():
    PLANNING_GROUP_ = "main_arm_SIM"
    rospy.init_node('get_ik_node',anonymous=True)
    # rospy.loginfo("node started")
    # print("planning group = ", PLANNING_GROUP_)
    base_pub = rospy.Publisher("base_pose",PoseStamped,queue_size=5)
    arm_pub = rospy.Publisher("arm_pose",PoseStamped,queue_size=5)
    
    # rospy.loginfo("Querying for FK")
    # gfk = GetFK('bvr_SIM/main_arm_SIM/gripper_manipulation_link', 'map')
    # resp = gfk.get_current_fk()
    # print("####################################")
    # print("fk solution", resp.pose_stamped[0].pose)
    # print("fk solution frame id", resp.pose_stamped[0].header.frame_id)
    # print("####################################")
    # print("\n\n\n")
    # print("####################################")
    
    # initilizing IK object
    ik_ob = GetIK(group=PLANNING_GROUP_)
    
    #initializing base position vector and direction vector 
    base_position_vec = np.array([0.0,0.0,0.0],dtype=np.float64)
    base_orientation_vec = np.array([1.0,1.0,0.0],dtype=np.float64)
    waypoint = np.hstack((base_position_vec,base_orientation_vec))
    
    #initilizing arm position in 3D space 
    arm_pose = PoseStamped()
    arm_pose.header.frame_id = 'map'
    
    #get base pose from position and direction vector 
    # if you have a valid base pose, use that instead as follows:
    # base_pos.pose = some_base_pose
    base_pos = getPoseFromBaseWaypoint(waypoint)
    base_pos.header = arm_pose.header
    base_pub.publish(base_pos)

    arm_pose.pose.position.x = 0.631212990079 + base_pos.pose.position.x
    arm_pose.pose.position.y = -0.1322317738 + base_pos.pose.position.y
    arm_pose.pose.position.z = 0.760689959739 + base_pos.pose.position.z
    arm_pose.pose.orientation.x = -0.00116679325101
    arm_pose.pose.orientation.y = 0.117537913367
    arm_pose.pose.orientation.z = 0.000954930466556
    arm_pose.pose.orientation.w = 0.993067251309

    arm_pub.publish(arm_pose)

    # print("point = ", arm_pose)
    
    response = ik_ob.get_ik(arm_pose, base_pos)
    print("ik response code", response.error_code)
    if(response.error_code.val == MoveItErrorCodes.SUCCESS):
        print("ik found\n")
    else:
        print("No solution found ")

    print("####################################")

    rate = rospy.Rate(3)
    while(not rospy.is_shutdown()):
        base_pub.publish(base_pos)
        arm_pub.publish(arm_pose)
        rate.sleep()



if __name__ == "__main__":
    main()