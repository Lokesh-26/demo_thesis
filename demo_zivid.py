import trimesh
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
from image_agnostic_segmentation.dounseen import utils, core
seg_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/segmentation/sam_vit_b_01ec64.pth'
cls_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/models/classification/vit_b_16_epoch_199_augment.pth'
k_path = '/media/gouda/3C448DDD448D99F2/datasets/br6d/cam_K.txt'
image_scale = 1
hope_dataset_gallery_path = '/media/gouda/3C448DDD448D99F2/segmentation/image_agnostic_segmentation/demo/objects_gallery'
cadmodel_path = '/media/gouda/3C448DDD448D99F2/datasets/br6d/models/obj_000003.ply'
def main():
    # create segmentation model
    segmentor = create_sam(seg_path, model_name='vit_b', device='cuda')
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
            sam_output = segmentor.generate(rgb_img)
            sam_masks, sam_bboxes = reformat_sam_output(sam_output)
            seg_img = utils.draw_segmented_image(rgb_img, sam_masks, sam_bboxes)
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

            # create DoUnseen classifier
            segments = utils.get_image_segments_from_binary_masks(rgb_img, sam_masks,
                                                                           sam_bboxes)  # get image segments from rgb image
            unseen_classifier = core.UnseenClassifier(
                model_path=cls_path,
                gallery_images=gallery_dict,
                gallery_buffered_path=None,
                augment_gallery=False,
                batch_size=32,
            )

            # find one object
            matched_query, score = unseen_classifier.find_object(segments, obj_name=search_obj_name, method="max")
            matched_query_ann_image = utils.draw_segmented_image(rgb_img,
                                                                          [sam_masks[matched_query]],
                                                                          [sam_bboxes[matched_query]], classes_predictions=[0],
                                                                          classes_names=["obj_000001"])
            matched_query_ann_image = cv2.cvtColor(matched_query_ann_image, cv2.COLOR_RGB2BGR)
            message = [('Classification done', 100),
                       ('Click enter to exit', 200)]
            # visualize classified image
            class_predictions, class_scores = unseen_classifier.classify_all_objects(segments,
                                                                                     threshold=0.5)
            filtered_class_predictions, filtered_masks, filtered_bboxes = utils.remove_unmatched_query_segments(
                class_predictions, sam_masks, sam_bboxes)

            classified_image = utils.draw_segmented_image(rgb_img, filtered_masks, filtered_bboxes,
                                                                   filtered_class_predictions,
                                                                   classes_names=None)
            classified_image = cv2.cvtColor(classified_image, cv2.COLOR_RGB2BGR)
            classified_visualize = classified_image.copy()
            add_multiple_text(classified_visualize, message)
            cv2.imshow('Demo', cv2.resize(classified_visualize, (0, 0), fx=image_scale, fy=image_scale))
            cv2.waitKey(0)

            cv2.destroyAllWindows()

            scaled_rgb, scaled_depth, K = downscale_rgb_depth_intrinsics(rgb_img, depth_img, K, shorter_side=400)
            mask = filtered_masks[0]
            # TODO: check if the mask is correct
            scaled_mask = process_mask(mask, scaled_rgb.shape[1], scaled_rgb.shape[0])
            pose = est.register(K=K, rgb=scaled_rgb, depth=scaled_depth, ob_mask=scaled_mask, iteration=5)
            pose = pose[0]
            # xyz_map = depth2xyzmap(depth_img, K)
            # valid = depth_img >= 0.001
            # pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            first_frame = False
        else:
            scaled_rgb, scaled_depth, K = downscale_rgb_depth_intrinsics(rgb_img, depth_img, K, shorter_side=400)
            pose = est.track_one(rgb=scaled_rgb, depth=scaled_depth, K=K, iteration=5)
        center_pose = pose@np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=rgb_img, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(rgb_img, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)


if __name__ == '__main__':
    main()