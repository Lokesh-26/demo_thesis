import trimesh
import zivid
import cv2
import numpy as np
import sys
import logging
import os
import glob
from PIL import Image
import open3d as o3d

import rospy
import roslaunch
import tf
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import TransformStamped

from zivid_camera.srv import *
from utils import *
import inflect
sys.path.append('/media/gouda/3C448DDD448D99F2/segmentation/')
sys.path.append('/media/gouda/3C448DDD448D99F2/segmentation/FoundationPose')
# import the module from the dir
from FoundationPose.estimater import *
from image_agnostic_segmentation.dounseen import utils, core
seg_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/segmentation/sam_vit_b_01ec64.pth'
cls_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/classification/vit_b_16_epoch_199_augment.pth'
k_path = '/media/gouda/3C448DDD448D99F2/datasets/br6d/cam_K.txt'
imgs_path = '/media/gouda/3C448DDD448D99F2/segmentation/demo_thesis/images'
image_scale = 0.5
hope_dataset_gallery_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/demo/objects_gallery'
cadmodel_path = '/media/gouda/3C448DDD448D99F2/datasets/br6d/models/obj_000003.ply'
obj_folder = 'big_klt'
bridge = CvBridge()


class ZividCamera:
    def __init__(self, sample_dir=None, frame_count=0):
        rospy.loginfo("Starting sample_capture_assistant.py")

        self.sample_dir = sample_dir
        self.frame_count = 0
        self.rgb_img = None
        self.depth_img = None

        #ca_suggest_settings_service = "/zivid_camera/capture_assistant/suggest_settings"
        #rospy.wait_for_service(ca_suggest_settings_service, 29.0)
        #self.capture_assistant_service = rospy.ServiceProxy(
        #    ca_suggest_settings_service, CaptureAssistantSuggestSettings
        #)

        self.capture_service = rospy.ServiceProxy("/zivid_camera/capture", Capture)

        settings_path = "/media/gouda/3C448DDD448D99F2/segmentation/demo_thesis/consumer_goods_fast.yml"
        self.load_settings_from_file_service = rospy.ServiceProxy(
            "/zivid_camera/load_settings_from_file", LoadSettingsFromFile
        )
        self.load_settings_from_file_service(settings_path)

        rospy.Subscriber("/zivid_camera/points/xyzrgba", PointCloud2, self.on_points)
        rospy.Subscriber('/zivid_camera/color/image_color', ImageMsg, self.rgb_callback)
        rospy.Subscriber('/zivid_camera/depth/image', ImageMsg, self.depth_callback)

    def capture_assistant_suggest_settings(self):
        max_capture_time = rospy.Duration.from_sec(10)
        rospy.loginfo(
            "Calling capture assistant service with max capture time = %.1f sec",
            max_capture_time.to_sec(),
        )
        self.capture_assistant_service(
            max_capture_time=max_capture_time,
            ambient_light_frequency=CaptureAssistantSuggestSettingsRequest.AMBIENT_LIGHT_FREQUENCY_NONE,
        )

    def capture(self):
        rospy.loginfo("Calling capture service")
        self.capture_service()

    def on_points(self, data):
        rospy.loginfo("PointCloud received")

    def rgb_callback(self, msg):
        self.rgb_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        rospy.loginfo("RGB callback triggered and image updated")

    def depth_callback(self, msg):
        depth_img = bridge.imgmsg_to_cv2(msg, "passthrough")
        depth_img = depth_img * 1000
        self.depth_img = depth_img.astype('uint16')
        rospy.loginfo("Depth callback triggered and image updated")

def main():
    rospy.init_node('data_collector', anonymous=True)
    # setup zivid camera
    zivid_cam = ZividCamera()
    # create segmentation model
    segmentor = create_sam(seg_path, model_name='vit_b', device='cuda')
    # make cv2 windows full screen
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # show teaser image
    # read a mp4 video and show it as a teaser gif
    teaser_video = cv2.VideoCapture('317561175-aa341004-5a15-4293-b3da-000471fd74ed.mp4')
    message = [('I can track any object that I have not seen before', 200),
               ('click enter', 300)]

    while True:
        # Read a frame from the video
        ret, frame = teaser_video.read()

        # If we've reached the end of the video, reset to the beginning
        if not ret:
            teaser_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Add text to the frame
        add_multiple_text(frame, message)

        # Resize and display the frame
        cv2.imshow('Demo', frame)

        # Wait briefly and check if 'Enter' key is pressed
        key = cv2.waitKey(30)  # Adjust delay as needed for playback speed
        if key == 13:  # Enter key
            break
    teaser_video.release()

    p = inflect.engine()  # to generate counting text: 1st, 2nd, 3rd, 4th, 5th, 6th
    # connect to the usb camera
    # cap = cv2.VideoCapture(0)  # 0 for laptop camera, 1 for usb camera
    search_obj_gallery_images = []
    # read the images from the imgs_path folder and append them to the search_obj_gallery_images
    for img in os.listdir(os.path.join(imgs_path, obj_folder)):
        img_path = os.path.join(imgs_path, obj_folder, img)
        img = cv2.imread(img_path)
        search_obj_gallery_images.append(img)

    # capture 6 images
    # for i in range(6):
    #     while True:
    #         ret, frame = cap.read()
    #         # crop the image to 640x480 image to 512x512 around the center
    #         frame = frame[80:560, 64:576]
    #         # resize the image to 256x256
    #         frame = cv2.resize(frame, (256, 256))
    #         message = [
    #             ('You need to capture 6 images of the object', 100),
    #             ('Click enter to capture the {} image'.format(p.number_to_words(p.ordinal(i + 1))), 200)
    #         ]
    #         captured_query_images = make_grid(search_obj_gallery_images + [frame], message)
    #         cv2.imshow('Demo', cv2.resize(captured_query_images, (0, 0), fx=image_scale, fy=image_scale))
    #         key = cv2.waitKey(50)
    #         # if enter is pressed, then add the captured image to the captured_query_image
    #         if key == 13:
    #             search_obj_gallery_images.append(frame)
    #             break
    # cap.release()

    # FoundationPose
    mesh = trimesh.load(cadmodel_path)
    mesh.apply_scale(0.001)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                         refiner=refiner, debug=0, glctx=glctx, debug_dir='/media/gouda/3C448DDD448D99F2/segmentation/FoundationPose/debug')
    logging.info("estimator initialization done")
    first_frame = True

    K = np.loadtxt(k_path).reshape(3, 3)
    while True:
        # frame = camera.capture(settings)
        zivid_cam.capture()
        # Wait until depth_img is not None
        start_time = rospy.Time.now()
        while zivid_cam.depth_img is None:
            if (rospy.Time.now() - start_time).to_sec() > 1.0:  # Timeout after 1 second
                rospy.logwarn("Depth image not received within timeout")
                break
            rospy.sleep(0.05)  # Check every 50 milliseconds

        if zivid_cam.depth_img is None:
            rospy.logwarn("Skipping this iteration due to missing depth image")
            continue
        rgb_img = zivid_cam.rgb_img
        depth_img = zivid_cam.depth_img
        # to meters
        scaled_rgb, scaled_depth, scaled_K = downscale_rgb_depth_intrinsics(rgb_img, depth_img, K, shorter_side=600)
        # float_rgb = scaled_rgb.astype(np.float64) / 255.0
        if first_frame:
            # add text to the rgb image
            message = [('Running segmentation', 100)]
            rgb_visualize = rgb_img.copy()
            add_multiple_text(rgb_visualize, message)
            # cv2.imshow('Demo', cv2.resize(rgb_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.imshow('Demo', rgb_visualize)
            cv2.waitKey(100)

            # segmentation
            sam_output = segmentor.generate(rgb_img)
            sam_masks, sam_bboxes = reformat_sam_output(sam_output)
            seg_img = utils.draw_segmented_image(rgb_img, sam_masks, sam_bboxes)
            # add text to the rgb image
            message = [('Segmentation of all objects is done', 100),
                       ('Click enter to search for your object', 200)]
            seg_visualize = seg_img.copy()
            add_multiple_text(seg_visualize, message)
            cv2.imshow('Demo', seg_visualize)
            cv2.waitKey(0)

            # add text to the rgb image
            message = [('Running classification', 100)]
            seg_visualize = seg_img.copy()
            add_multiple_text(seg_visualize, message)
            cv2.imshow('Demo', seg_visualize)
            cv2.waitKey(100)

            # classification
            unseen_classifier = core.UnseenClassifier(
                model_path=cls_path,
                gallery_images=None,
                gallery_buffered_path=None,
                augment_gallery=False,
                batch_size=32,
            )

            unseen_classifier.update_gallery(imgs_path)
            # create DoUnseen classifier
            segments = utils.get_image_segments_from_binary_masks(rgb_img, sam_masks,
                                                                           sam_bboxes)  # get image segments from rgb image
            # find one object
            matched_query, score = unseen_classifier.find_object(segments, obj_name=obj_folder, method="max")
            matched_query_ann_image = utils.draw_segmented_image(rgb_img,
                                                                          [sam_masks[matched_query]],
                                                                          [sam_bboxes[matched_query]], classes_predictions=[0],
                                                                          classes_names=["your_object"])
            matched_query_ann_image = cv2.cvtColor(matched_query_ann_image, cv2.COLOR_RGB2BGR)
            message = [('Classification done', 100),
                       ('Click enter to track the object', 200)]
            # visualize classified image
            class_predictions, class_scores = unseen_classifier.classify_all_objects(segments,
                                                                                     threshold=0.5)
            filtered_class_predictions, filtered_masks, filtered_bboxes = utils.remove_unmatched_query_segments(
                class_predictions, sam_masks, sam_bboxes)

            classified_image = utils.draw_segmented_image(rgb_img, filtered_masks, filtered_bboxes,
                                                                   filtered_class_predictions,
                                                                   classes_names=None)
            # classified_image = cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)
            classified_visualize = classified_image.copy()
            add_multiple_text(classified_visualize, message)
            cv2.imshow('Demo', classified_visualize)
            cv2.waitKey(0)

            # # cv2.destroyAllWindows()

            mask = filtered_masks[0]
            scaled_mask = process_mask(mask, scaled_rgb.shape[1], scaled_rgb.shape[0])
            scaled_mask = scaled_mask.astype(bool)
            # initialize the pose estimation
            pose = est.register(K=scaled_K, rgb=scaled_rgb, depth=scaled_depth, ob_mask=scaled_mask, iteration=5)
            m = mesh.copy()
            m.apply_transform(pose)
            m.export('model_tf.obj')
            xyz_map = depth2xyzmap(scaled_depth, scaled_K)
            valid = scaled_depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], scaled_rgb[valid])
            o3d.io.write_point_cloud('scene_complete.ply', pcd)
            first_frame = False
        else:
            pose = est.track_one(rgb=scaled_rgb, depth=scaled_depth, K=scaled_K, iteration=5)
        center_pose = pose@np.linalg.inv(to_origin)
        scaled_rgb = cv2.cvtColor(scaled_rgb, cv2.COLOR_BGR2RGB)
        # scaled_rgb_visualize = scaled_rgb.copy()
        vis = draw_posed_3d_box(scaled_K, img=scaled_rgb, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(scaled_rgb, ob_in_cam=center_pose, scale=0.1, K=scaled_K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('Demo', vis[...,::-1])
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
    rospy.spin()  # This keeps the node active to receive callbacks
