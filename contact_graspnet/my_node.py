#!/usr/bin/python3
import os
import sys
import argparse
import cv2

from data import depth2pc

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo
from contact_graspnet_planner.srv import ContactGraspNetPlanner, ContactGraspNetPlannerResponse
from contact_graspnet_planner.msg import ContactGraspVect
from contact_graspnet_planner.msg import ContactGrasp

# import glob
import copy as copy_module

# vizual
from matplotlib import pyplot as plt

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import load_available_input_data
from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from std_srvs.srv import Trigger, TriggerResponse

import numpy as np
from scipy.spatial.transform import Rotation as R
import threading

lock = threading.Lock()
# freq = 100
# conf_thresh = 0.8
# min_dist_thresh = 0.2


class ImageListener:

    def __init__(self):
        self.im_ros = None
        self.im = None
        self.depth = None
        self.depth_ros = None
        self.depth_crp = None
        self.depth_crp_ros = None
        self.depth_encoding = None
        self.camera_info_ros = None
        # self.seg_ros = None
        # self.seg = None
        self.segmask = Image()
        self.resp = ContactGraspNetPlannerResponse()
        self.pc_full = None
        self.flag = 0
        self.camera_matrix_K = None
        self.cv_brdg = CvBridge()
        self.grasp_msg = ContactGraspVect() 

        # initialize a node
        rospy.init_node("sod", log_level=rospy.INFO)
        self.rgb_pub = rospy.Publisher('/rgb', Image, queue_size=10)

        self.depth_pub = rospy.Publisher('/depth', Image, queue_size=10)

        # publish grasp position and orientation
        self.ans = rospy.Publisher('/answer', ContactGraspVect, queue_size=10)

        rgb_sub = message_filters.Subscriber('/camera/color/image_raw',
                                             Image, queue_size=10)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                               Image, queue_size=10)
        camera_sub = message_filters.Subscriber('/camera/depth/camera_info',
                                               CameraInfo, queue_size=10)
        depth_crop_sub = rospy.Subscriber('/depth_masked',
                                               Image, self.callback_dep, queue_size=10)

        my_service = rospy.Service('/get_grasps', Trigger, self.trigger_response)
        
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, camera_sub], 10, 0.1)
        ts.registerCallback(self.callback_rgbd)


    def trigger_response(self,request):

        self.start_srv()
        self.flag = 1

        return TriggerResponse(
            success=True,
            message="Hey, roger that; we'll be right there!"
        )


    def ros_depthmsg_to_pc(self, ros_depth, K, ros_rgb, encoding='passthrough'):
        depth_norm = None

        rgb = self.cv_brdg.imgmsg_to_cv2(ros_rgb,encoding)

        depth_numpy = self.cv_brdg.imgmsg_to_cv2(ros_depth,encoding)
        depth_norm = cv2.normalize(depth_numpy, depth_norm, 0, 1, cv2.NORM_MINMAX)
        return depth2pc(depth_norm, K, rgb)


    def depthmsg_to_depthnorm(self, ros_depth, encoding='passthrough'):
        depth_norm = None

        depth_numpy = self.cv_brdg.imgmsg_to_cv2(ros_depth,encoding)
        depth_norm = cv2.normalize(depth_numpy, depth_norm, 0, 1, cv2.NORM_MINMAX)
        return depth_norm



    def callback_rgbd(self, rgb, depth, cam_K):


        self.camera_matrix_K = np.array(cam_K.K).reshape((3, 3))
        # img_pc, pc_colors = self.ros_depthmsg_to_pc(depth, self.camera_matrix_K , rgb)
        # print("img_pc shape:")
        # print(img_pc.shape)


        self.depth_encoding = depth.encoding
        if depth.encoding == '32FC1':
            depth_cv = self.cv_brdg.imgmsg_to_cv2(depth)
        elif depth.encoding == '16UC1':
            depth_cv = self.cv_brdg.imgmsg_to_cv2(
                depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = self.cv_brdg.imgmsg_to_cv2(rgb)

        with lock:
            self.im_ros = rgb
            self.im = im.copy()
            self.depth_ros = depth
            self.depth = depth_cv.copy()
            self.camera_info_ros = cam_K
            self.segmask = copy_module.deepcopy(rgb)
            self.segmask.encoding = '8UC1'
            self.segmask.height = 1
            self.segmask.width = 1
            self.segmask.step = 1
            self.segmask.data = [1]
            
    
    def callback_dep(self, depth_crop):

        #seg = self.cv_brdg.imgmsg_to_cv2(segmask)

        with lock:
            #self.seg=seg.copy()
            self.depth_crp_ros = depth_crop

    # def callback_seg(self, segmask):

    #     #seg = self.cv_brdg.imgmsg_to_cv2(segmask)
    #     print('segmask')
    #     print(segmask)
    #     with lock:
    #         #self.seg=seg.copy()
    #         self.seg_ros = segmask

    
    def run_proc(self):

        if not (self.im_ros == None):
            if not (self.depth_ros == None):

                self.rgb_pub.publish(self.im_ros)
                self.depth_pub.publish(self.depth_ros)
            else: 
                print('im and depth are None')
        else:
            print('depth and im are None')


    def start_srv(self):

        # if (self.im_ros == None and self.depth_ros == None and self.seg_ros == None):
        if (self.im_ros == None and self.depth_crp_ros == None):
            print("No valid msg")
            return

        # request service to server
        rospy.loginfo('Start grasp_planner_client')
        service_name = 'grasp_planner'
        rospy.loginfo('Wait for the grasp_planner_server')
        rospy.wait_for_service(service_name)        
            
        try:
            rospy.loginfo("Request Contact-GraspNet grasp planning")
            grasp_planner = rospy.ServiceProxy(service_name, ContactGraspNetPlanner)
            # rospy.loginfo("smth") # works 
            self.resp = grasp_planner(
                    self.im_ros,
                    self.depth_crp_ros,
                    self.camera_info_ros,
                    self.segmask
                    )
            rospy.loginfo("Get {} grasps from the server.".format(len(self.resp.grasps)))
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: {}".format(e))

    def make_grasp_msg(self, se3, score, contact_point, ind):
        msg = ContactGrasp()
        msg.score = score
        msg.id = ind
        msg.contact_point.x = contact_point[0]
        msg.contact_point.y = contact_point[1]
        msg.contact_point.z = contact_point[2]


        r = R.from_matrix(se3[0:3, 0:3])
        quat = r.as_quat() # quat shape (4,)

        point = se3[0:3, 3]

        msg.pose.position.x = point[0]
        msg.pose.position.y = point[1]
        msg.pose.position.z = point[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        return msg
    

if __name__ == '__main__':

    # cv_brdg = CvBridge()     
    # ros_img_cv = cv_brdg.imgmsg_to_cv2(self.im,"rgb8")
    # show_image(ros_img_cv, None)


    listener = ImageListener()
    rate = rospy.Rate(100)
    print('main')
    global_config = config_utils.load_config(
        '/home/diana/ros_ws/src/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001/config.yaml', batch_size=1, arg_configs=[]
        )
    grasp_estimator = GraspEstimator(global_config)
    # print(listener.flag)

    while not rospy.is_shutdown():

        if  ((listener.flag==1)&(listener.im_ros!=None)&(listener.depth_ros!=None)):
            # prepare
            # get depth from msg
            # depth_normalized = listener.depthmsg_to_depthnorm(listener.depth)
            # img_16uc1 = listener.cv_brdg.imgmsg_to_cv2(listener.depth_ros, desired_encoding='16UC1')
            # depth_normalized = (img_16uc1.astype(np.float64) / (2**16)).astype(np.float32)
            # print("depth numpy")
            # print(depth_normalized)
            # print("depth mean")
            # print(np.mean(depth_normalized))
            # from ros msg to array
            # ros_img_cv = listener.cv_brdg.imgmsg_to_cv2(listener.im_ros,"rgb8")

            # convert data to point cloud
            pc_full, pc_segments, pc_color = grasp_estimator.extract_point_clouds(
                depth=listener.depth,
                K=listener.camera_matrix_K,
                rgb=listener.im,
                skip_border_objects=False,
                z_range=[0.2, 1.8],
                )
            # pc_full,pc_color = listener.ros_depthmsg_to_pc(listener.depth, listener.camera_matrix_K, listener.im)    


            # visulaize grasp
            pred_grasps_cam = {}
            scores = {}
            contact_pts = {}

            id_list = []
            grasp_list = []
            scores_list = []
            contact_pts_list = []
            for grasp in listener.resp.grasps:
                instance_id = grasp.id
                pose_msg = grasp.pose
                score = grasp.score
                contact_pt_msg = grasp.contact_point

                # get transform matrix
                tf_mat = np.zeros((4, 4), dtype=np.float64)
                quat = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
                rot_mat = R.from_quat(quat)
                tf_mat[0:3, 0:3] = rot_mat.as_matrix()
                tf_mat[0, 3] = pose_msg.position.x
                tf_mat[1, 3] = pose_msg.position.y
                tf_mat[2, 3] = pose_msg.position.z

                # get contact point as numpy
                contact_pt = np.array([contact_pt_msg.x, contact_pt_msg.y, contact_pt_msg.z])

                # append to list
                id_list.append(instance_id)
                grasp_list.append(tf_mat)
                scores_list.append(score)
                contact_pts_list.append(contact_pt)

            # convert list to numpy array
            id_list = np.array(id_list)
            grasp_list = np.array(grasp_list)
            scores_list = np.array(scores_list)
            contact_pts_list = np.array(contact_pts_list)

            # put on the dictionary
            for instance_id in id_list:
                indices = np.where(id_list == instance_id)[0]
                pred_grasps_cam[instance_id] = grasp_list[indices]
                scores[instance_id] = scores_list[indices]
                contact_pts[instance_id] = contact_pts_list[indices]

            # make ros msg of top 5 highest score grasps (pose,score,contact points, id)
            for i,k in enumerate(pred_grasps_cam):
                largest_scores_ind = np.argsort(scores[k])
                top5 = largest_scores_ind[-5:]
                for j in range(len(top5)):
                    listener.grasp_msg.grasps_vect.append(
                        listener.make_grasp_msg(
                            pred_grasps_cam[k][top5[j]],
                            scores[k][top5[j]],
                            contact_pts[k][top5[j]],
                            top5[j]
                            )
                        )

            listener.ans.publish(listener.grasp_msg)
            # clear old data
            listener.grasp_msg = ContactGraspVect()

            show_image(listener.im, None)
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_color)
            listener.flag = 0
    rate.sleep()
    