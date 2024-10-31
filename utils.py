import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

if not torch.cuda.is_available():
    print("No GPU found, this package is not optimized for CPU. Exiting ...")
    exit()
else:
    device = torch.device("cuda")
def make_grid(captured_images, message):
    images = captured_images.copy()
    # if list if images is less than 6 images, then fill the list till 6 images with white images
    if len(images) < 6:
        # make an empty white image
        images.extend([np.zeros((256,256,3), np.uint8) for i in range(6-len(images))])
    # resize all images to 256x256
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], (256,256))
    nrow=3
    h_padding = int((1944-(256*3))/3/2)
    v_padding = 50
    # adding a white frame around each image with padding size
    for i in range(len(images)):
        images[i] = cv2.copyMakeBorder(images[i], v_padding, v_padding, h_padding, h_padding, cv2.BORDER_CONSTANT, value=[255,255,255])
    # stack images in a grid with some padding in between
    grid = []
    for i in range(0, len(images), nrow):
        row = []
        for j in range(nrow):
            if i+j < len(images):
                row.append(images[i+j])
            else:
                row.append(np.zeros_like(images[0]))
        grid.append(np.hstack(row))
    grid = np.vstack(grid)

    # append a message to the grid's bottom
    msg_img_w = 1944
    msg_img_h = 1200 - ((256+v_padding+v_padding)*2)
    # make msg image with white background
    msg_img = np.zeros((msg_img_h, msg_img_w, 3), np.uint8)
    msg_img[:] = (255,255,255)
    # add multi line text to the msg image
    add_multiple_text(msg_img, message)
    # append msg image to the grid
    grid = np.vstack([grid, msg_img])

    return grid

def add_multiple_text(img, message):
    for line, y in message:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)
        text_w = text_size[0][0]
        text_h = text_size[0][1]
        text_x = int((img.shape[1]-text_w)/2)
        text_y = y
        cv2.putText(img, line, (text_x, text_y), font, font_scale, (256,175,0), font_thickness, cv2.LINE_AA)
        
def create_sam(sam_model_path, model_name='vit_b', device='cuda'):
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.float16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam = sam_model_registry[model_name](sam_model_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=50,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing,

    )
    return mask_generator

def reformat_sam_output(sam_output):
    masks = []
    bboxes = []
    for idx in range(len(sam_output)):
        mask = sam_output[idx]['segmentation']
        bbox = sam_output[idx]['bbox']
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # change bboxes from xywh to xyxy
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        masks.append(mask)
        bboxes.append(bbox)
    return masks, bboxes


def downscale_rgb_depth_intrinsics(rgb, depth, K, shorter_side=400):
    """
    Downscale RGB and depth images along with camera intrinsics.

    Parameters:
    rgb (numpy.ndarray): The RGB image.
    depth (numpy.ndarray): The depth image.
    K (numpy.ndarray): Camera intrinsic matrix.
    shorter_side (int): The desired size for the shorter side of the image.

    Returns:
    downscaled_rgb (numpy.ndarray): Downscaled RGB image.
    downscaled_depth (numpy.ndarray): Downscaled depth image.
    new_K (numpy.ndarray): Adjusted camera intrinsic matrix.
    """
    # Get the original dimensions
    original_height, original_width = rgb.shape[:2]

    # Calculate downscale factor based on the shorter side
    downscale_factor = shorter_side / min(original_height, original_width)

    # Downscale dimensions
    new_height = int(original_height * downscale_factor)
    new_width = int(original_width * downscale_factor)

    # Downscale the RGB and depth images
    downscaled_rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    downscaled_depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # Adjust the intrinsic matrix
    new_K = K.copy()
    new_K[:2] *= downscale_factor

    return downscaled_rgb, downscaled_depth, new_K


def process_mask(mask, new_width, new_height):
    """
    Process and resize a mask to a binary format.

    Parameters:
    mask (numpy.ndarray): The RGB or single-channel mask image.
    new_width (int): The width to resize the mask to.
    new_height (int): The height to resize the mask to.

    Returns:
    downscaled_mask (numpy.ndarray): Downscaled binary mask.
    """
    # If mask has multiple channels, select one with non-zero content
    if len(mask.shape) == 3:
        for c in range(3):
            if mask[..., c].sum() > 0:
                mask = mask[..., c]
                break

    # Resize and convert to binary mask
    downscaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # downscaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(bool).astype(
    #     np.uint8)
    return downscaled_mask