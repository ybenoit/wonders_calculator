import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def create_base_image(im, box, max_num_pixels_above=40, max_num_pixels_below=10):
    pixel_above = np.random.randint(max_num_pixels_above + max_num_pixels_below + 1) - max_num_pixels_below
    pixel_below = np.random.randint(max_num_pixels_above + max_num_pixels_below + 1) - max_num_pixels_below
    pixel_left = np.random.randint(max_num_pixels_above + max_num_pixels_below + 1) - max_num_pixels_below
    pixel_right = np.random.randint(max_num_pixels_above + max_num_pixels_below + 1) - max_num_pixels_below

    im_shape = im.shape
    base_image = cv2.cvtColor(im[
                              np.max([box[1]-pixel_above, 0]):np.min([box[3]+pixel_below, im_shape[0]]),
                              np.max([box[0]-pixel_left, 0]):np.min([box[2]+pixel_right, im_shape[1]]), :], cv2.COLOR_BGR2RGB)
    base_image_shape = base_image.shape
    return (base_image,
            np.max([pixel_left, 0]),
            np.max([pixel_above, 0]),
            base_image_shape[1] - np.max([pixel_right, 0]),
            base_image_shape[0] - np.max([pixel_below, 0]))


def find_suffix(name, base_img_name_suffix):
    index_match_start = name.find(base_img_name_suffix)
    index_match_end= name.find(".")
    if index_match_start != -1:
        suffix = name[index_match_start+len(base_img_name_suffix):index_match_end]
        if suffix.isdigit():
            return int(suffix) + 1


def save_base_images_with_labels_from_box(im, box, class_name, num_images_to_generate=10,
                                          save_dir="/home/yoann/dev/object_detection/data/"):

    # Find max image index saved for the same class
    list_dir = os.listdir("/home/yoann/dev/object_detection/data/wonders_base_images")
    base_img_name_suffix = "base_img_%s_" % (class_name)
    if len(list_dir) == 0:
        max_image_index = 0
    else:
        max_image_index = np.max(map(lambda name: find_suffix(name, base_img_name_suffix), list_dir))
        if max_image_index is None:
            max_image_index = 0

    for i in range(num_images_to_generate):
        # Create base image
        (base_img, new_box_0, new_box_1, new_box_2, new_box_3) = create_base_image(im, box)

        # Generate save path
        save_path = save_dir + "wonders_base_images/" + base_img_name_suffix + "%s.png" % (i + max_image_index)

        # Save image
        plt.imsave(save_path, base_img)

        # Append box position
        with open(save_dir + "base_images_box.txt", 'a') as f:
            f.write("%s,%s,%s,%s,%s,%s\n" % (save_path, new_box_0, new_box_1, new_box_2, new_box_3, class_name))
