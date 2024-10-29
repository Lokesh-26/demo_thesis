import cv2
import numpy as np


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