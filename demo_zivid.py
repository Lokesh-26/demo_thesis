import zivid
import cv2
import numpy as np
import sys
import logging
import os
import glob
from PIL import Image


from utils import *
import inflect
sys.path.append('/media/gouda/3C448DDD448D99F2/segmentation/')
sys.path.append('/media/gouda/3C448DDD448D99F2/segmentation/FoundationPose')
# import the module from the dir
from FoundationPose.estimater import *
from image_agnostic_segmentation.dounseen import UnseenClassifier, UnseenSegment, draw_segmented_image
seg_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/segmentation/segmentation_mask_rcnn.pth'
cls_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/classification/classification_vit_b_16_ctl.pth'
image_scale = 1
hope_dataset_gallery_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/demo/objects_gallery'

def main():
    # initialize the DoUnseen
    segmentor = UnseenSegment(method='maskrcnn', maskrcnn_model_path=seg_path)
    zero_shot_classifier = UnseenClassifier(model_path=cls_path)
    # make cv2 windows full screen
    cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Demo', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # show teaser image
    teaser_img = cv2.imread('teaser.png')  # image in the same directory as this script
    # add text to the teaser image
    message = [('I can segment any new object from few images', 200),
               ('Wanna play? click enter', 300)]
    add_multiple_text(teaser_img, message)
    # resize teaser image to half
    cv2.imshow('Demo', cv2.resize(teaser_img, (0,0), fx=image_scale, fy=image_scale))
    cv2.waitKey(0)

    ## acquire query images from usb camera
    # show an example for the user how to capture query images
    obj_000001_query_images = glob.glob(os.path.join(hope_dataset_gallery_path, 'obj_000001', '*.jpg'))  # 6 images
    obj_000001_query_images = [cv2.imread(img) for img in obj_000001_query_images]
    # stack image in a grid
    message = [('Grab any object you have with you, You will capture few images of it', 100),
               ('This is an example of how to capture query images', 200),
               ('Click enter to start capturing', 300)]
    search_obj_gallery_images = make_grid(obj_000001_query_images, message)
    cv2.imshow('Demo', cv2.resize(search_obj_gallery_images, (0,0), fx=image_scale, fy=image_scale))
    cv2.waitKey(0)

    p = inflect.engine()  # to generate counting text: 1st, 2nd, 3rd, 4th, 5th, 6th
    # connect to the usb camera
    cap = cv2.VideoCapture(0)  # 0 for laptop camera, 1 for usb camera
    search_obj_gallery_images = []
    # capture 6 images
    for i in range(6):
        while True:
            ret, frame = cap.read()
            # crop the image to 640x480 image to 512x512 around the center
            frame = frame[80:560, 64:576]
            # resize the image to 256x256
            frame = cv2.resize(frame, (256, 256))
            message = [
                ('You need to capture 6 images of the object', 100),
                ('Click enter to capture the {} image'.format(p.number_to_words(p.ordinal(i + 1))), 200)
            ]
            captured_query_images = make_grid(search_obj_gallery_images + [frame], message)
            cv2.imshow('Demo', cv2.resize(captured_query_images, (0, 0), fx=image_scale, fy=image_scale))
            key = cv2.waitKey(50)
            # if enter is pressed, then add the captured image to the captured_query_image
            if key == 13:
                search_obj_gallery_images.append(frame)
                break
    cap.release()

    app = zivid.Application()
    camera = app.connect_camera()
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    settings.load('/media/gouda/3C448DDD448D99F2/segmentation/demo_thesis/consumer_goods_fast.yml')
    rgb_gallery = []

    # FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # est = FoundationPose(model_pts=None, model_normals=None, scorer=scorer,
    #                      refiner=refiner, debug=4, glctx=glctx, debug_dir='/media/gouda/3C448DDD448D99F2/segmentation/FoundationPose/debug')
    logging.info("estimator initialization done")
    first_frame = True

    K = np.array([[1.778810058593750000e+03, 0.0, 9.679315795898438000e+02],
                  [0.0, 1.778870361328125000e+03, 5.724088134765625000e+02],
                  [0.0, 0.0, 1.0]])
    while True:
        frame = camera.capture(settings)
        rgb_img = frame.point_cloud().copy_data('rgba')
        # remove alpha channel
        rgb_img = rgb_img[:,:,:3]
        # convert rgb to bgr
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        depth_img = frame.point_cloud().copy_data('z')
        if first_frame:
            # add text to the rgb image
            message = [('Running segmentation', 100)]
            rgb_visualize = rgb_img.copy()
            add_multiple_text(rgb_visualize, message)
            cv2.imshow('Demo', cv2.resize(rgb_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.waitKey(100)

            # segmentation
            seg_predictions = segmentor.segment_image(rgb_img)
            seg_img = draw_segmented_image(rgb_img, seg_predictions)
            # add text to the rgb image
            message = [('Segmentation of all objects is done', 100),
                       ('Click enter to search for your object', 200)]
            seg_visualize = seg_img.copy()
            add_multiple_text(seg_visualize, message)
            cv2.imshow('Demo', cv2.resize(seg_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.waitKey(0)

            # add text to the rgb image
            message = [('Running classification', 100)]
            seg_visualize = seg_img.copy()
            add_multiple_text(seg_visualize, message)
            cv2.imshow('Demo', cv2.resize(seg_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.waitKey(100)
            # classification
            # load gallery images
            gallery_images_path = '/media/gouda/3C448DDD448D99F2/segmentation/demo_thesis/gallery_real_resized_256'
            # create dictionary of gallery images. Each item in the dictionary is a list of PIL images of the same object
            # list object in gallery folder
            gallery_objects = os.listdir(gallery_images_path)
            gallery_dict = {}
            for obj in gallery_objects:
                gallery_dict[obj] = []
                # list images in each object folder
                obj_images = glob.glob(os.path.join(gallery_images_path, obj, '*.jpg'))
                for img in obj_images:
                    gallery_dict[obj].append(Image.open(img))

            # append search object images to the gallery dictionary
            search_obj_name = 'your_object'
            gallery_dict[search_obj_name] = []
            for img in search_obj_gallery_images:
                gallery_dict[search_obj_name].append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

            zero_shot_classifier.update_gallery(gallery_dict)
            class_predictions = zero_shot_classifier.find_object(rgb_img, seg_predictions, obj_name=search_obj_name)
            classified_image = draw_segmented_image(rgb_img, class_predictions, classes=[search_obj_name])
            # add text to the classified image
            message = [('Classification done', 100),
                       ('Click enter to exit', 200)]
            # visualize classified image
            classified_visualize = classified_image.copy()
            add_multiple_text(classified_visualize, message)
            cv2.imshow('Demo', cv2.resize(classified_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.waitKey(0)

            cv2.destroyAllWindows()

            mask = None
            pose = est.register(K=K, rgb=rgb_img, depth=depth_img, ob_mask=mask, iteration=5)
            pose = pose[0]
            # xyz_map = depth2xyzmap(depth_img, K)
            # valid = depth_img >= 0.001
            # pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            first_frame = False
        else:
            pose = est.track_one(rgb=rgb_img, depth=depth_img, K=K, iteration=5)
    center_pose = pose@np.linalg.inv(to_origin)
    vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[...,::-1])
    cv2.waitKey(1)


    # Display the final captured RGB and depth images
    cv2.imshow('RGB Image', rgb_img)
    cv2.imshow('Depth Image', depth_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()