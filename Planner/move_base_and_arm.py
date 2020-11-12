import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from get_ik import GetFK, GetIK
from geometry_msgs.msg import PoseStamped, Pose

class moveMobileManipulator:
    '''
    implementation of moving the base and manipulator arm.
    '''
    def __init__(self,moveit_commander):
        self.PLANNING_GROUP = "main_arm_SIM"
        self.gripper_link = 'bvr_SIM/main_arm_SIM/gripper_manipulation_link'
        self.base_link = 'bvr_SIM/base_link'
        # self.Ik_ob = GetIK(group=self.PLANNING_GROUP)
        self.ee_goal_pose = PoseStamped()
        self.ee_goal_pose.header.frame_id = 'map'
        self.moveit_commander = moveit_commander
        self.robot = self.moveit_commander.RobotCommander(robot_description='bvr_SIM/robot_description')
        print("##### 1")
        self.scene = self.moveit_commander.PlanningSceneInterface()
        print("##### 2")
        self.group_name = self.PLANNING_GROUP
        print("##### 3")
        self.group = self.moveit_commander.MoveGroupCommander(self.group_name)
        print("##### 4")

    def move_arm(self, pose_goal):
        # We can get the name of the reference frame for this robot:
        planning_frame = self.group.get_planning_frame()
        print("============ Reference frame: %s" % planning_frame)
        # We can also print the name of the end-effector link for this group:
        eef_link = self.group.get_end_effector_link()
        print( "============ End effector: %s" % eef_link)
        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print ("============ Robot Groups:", self.robot.get_group_names())
        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print( "============ Printing robot state")
        print(self.robot.get_current_state())
        print("")

        self.group.set_pose_target(pose_goal)
        plan = self.group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        self.group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        self.group.clear_pose_targets()


def main():
    rospy.init_node('moveit_move_base_and_arm',anonymous=True)
    pose_goal = Pose()
    pose_goal.orientation.w = 1.0
    pose_goal.position.x = 0.4
    pose_goal.position.y = 0.1
    pose_goal.position.z = 0.4
    moveit_commander.roscpp_initialize(sys.argv)
    move_ = moveMobileManipulator(moveit_commander)
    move_.move_arm(pose_goal)


if __name__ == '__main__':
    main()