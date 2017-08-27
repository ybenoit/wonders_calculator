import cv2
import numpy as np
import os
import pandas as pd
import scipy.misc
import argparse


def create_fake_train_image(background_images_path,
                            base_images, base_images_boxes, base_images_classes,
                            nb_base_images=5, train_image_height=500, train_image_width=1000):
    """
    Creates a fake training image by putting some base sub images randomly into a fake image

    background_images_path : List of background images paths
    base_images : List of base images from which to pick in order to create a fake training image
    base_images_boxes : List the base images' associated box coordinates
    base_images_classes : List of the base images' associated class
    nb_base_images : Number of base images to put in the fake training image
    train_image_height : Height of the generated fake training image
    train_image_width : Width of the generated fake training image
    """
    # TODO : Add random background
    # TODO : Add random base image size
    # TODO : Add random transformations (rotation, crop)

    # Initialize training image with a random background image
    if np.random.random() > 0.6:
        train_image = np.zeros([train_image_height, train_image_width, 3], dtype=np.uint8)
        train_image[:, :, 0] = np.random.randint(255)
        train_image[:, :, 1] = np.random.randint(255)
        train_image[:, :, 2] = np.random.randint(255)
    # Initialize training image with a background image
    else:
        train_image = cv2.imread(background_images_path[np.random.randint(len(background_images_path))])
    train_height, train_width, _ = train_image.shape

    # Randomly pick indexes of the base images to incorporate into the training image
    base_images_indexes = np.random.randint(0, len(base_images), nb_base_images)

    # Initialize the boxes and classes representing the training image (the outputs to predict)
    train_image_boxes = []
    train_image_classes = []

    # Iteratively putting base images on the training image
    for base_image_index in base_images_indexes:

        # Select the base image to incorporate and its information
        base_image = base_images[base_image_index]
        base_image_height, base_image_width, _ = base_image.shape
        base_images_box = base_images_boxes[base_image_index]

        # Add the base image class to the training image classes list
        train_image_classes.append(base_images_classes[base_image_index])

        # If the shape of the base image is to high, reduce it. Otherwise stay with the base image as is
        x_ratio = 1. * base_image_width / train_width
        y_ratio = 1. * base_image_height / train_height
        max_ratio = np.max([x_ratio, y_ratio])

        min_ratio = np.random.uniform(0.01, 0.15)

        if max_ratio > min_ratio:
            coeff_reduce = min_ratio / max_ratio

            base_image_reshape = cv2.resize(base_image,
                                            (0, 0),
                                            fx=coeff_reduce,
                                            fy=coeff_reduce,
                                            interpolation=cv2.INTER_CUBIC)
            base_image_box_reshape = [int(base_images_box[i] * coeff_reduce) for i in range(4)]

        else:
            base_image_reshape = base_image
            base_image_box_reshape = base_images_box

        base_image_reshape_height, base_image_reshape_width, _ = base_image_reshape.shape

        # Random define the base image position in the training image
        x_base_image = np.random.randint(train_width - base_image_reshape_width)
        y_base_image = np.random.randint(train_height - base_image_reshape_height)

        base_image_box_reshape[0] += x_base_image
        base_image_box_reshape[1] += y_base_image
        base_image_box_reshape[2] += x_base_image
        base_image_box_reshape[3] += y_base_image
        train_image_boxes.append(base_image_box_reshape)

        # Incorporate the base image into the training image in the right position
        train_image[y_base_image:y_base_image + base_image_reshape_height,
        x_base_image:x_base_image + base_image_reshape_width] = base_image_reshape

    return train_image, train_image_boxes, train_image_classes


def find_suffix(name, base_img_name_suffix):
    index_match_start = name.find(base_img_name_suffix)
    index_match_end = name.find(".")
    if index_match_start != -1:
        suffix = name[index_match_start + len(base_img_name_suffix):index_match_end]
        if suffix.isdigit():
            return int(suffix) + 1


def save_train_images_with_labels_and_boxes(num_images_to_generate,
                                            artificial_training_images_save_path,
                                            artificial_training_images_box_save_path,
                                            class_list,
                                            base_images_boxes,
                                            base_images,
                                            background_images_path):

    ARTIFICIAL_TRAINING_IMAGES_SAVE_PATH = artificial_training_images_save_path
    ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH = artificial_training_images_box_save_path

    # Find max image index saved for the same class
    list_dir = os.listdir(ARTIFICIAL_TRAINING_IMAGES_SAVE_PATH)
    train_img_name_suffix = "train_img_"
    if len(list_dir) == 0:
        max_image_index = 0
    else:
        max_image_index = np.max(map(lambda name: find_suffix(name, train_img_name_suffix), list_dir))
        if max_image_index is None:
            max_image_index = 0

    for i in range(num_images_to_generate):
        print("Generating image %s" % (i))

        train_image_height, train_image_width = np.random.randint(300, 2000, 2)
        nb_sub_images = np.random.randint(3, 25)

        # Create base image
        my_img, my_img_boxes, my_img_classes = create_fake_train_image(
            background_images_path=background_images_path,
            base_images=base_images,
            base_images_boxes=base_images_boxes,
            base_images_classes=class_list,
            nb_base_images=nb_sub_images,
            train_image_height=train_image_height,
            train_image_width=train_image_width)

        # Generate save path
        image_name = train_img_name_suffix + "%s.jpg" % (i + max_image_index)
        img_save_path = os.path.join(ARTIFICIAL_TRAINING_IMAGES_SAVE_PATH, image_name)

        # Save image
        # plt.imsave(img_save_path, cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))
        scipy.misc.imsave(img_save_path, cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB))

        # Check if save file already exists
        if not os.path.exists(ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH):
            with open(ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH, 'a') as f:
                f.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

        # Append box position
        with open(ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH, 'a') as f:
            my_img_height, my_img_width, _ = my_img.shape
            for j in range(len(my_img_boxes)):
                base_img_box = my_img_boxes[j]
                base_img_class = my_img_classes[j]
                f.write("%s,%s,%s,%s,%s,%s,%s,%s\n" % (image_name, my_img_width, my_img_height,
                                                       base_img_class, base_img_box[0], base_img_box[1],
                                                       base_img_box[2], base_img_box[3]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_images", type=int, default=100,
                        help="How many traiing images to generate.")
    parser.add_argument("--base_images_path")
    parser.add_argument("--base_images_box_path")
    parser.add_argument("--background_images_path")
    parser.add_argument("--artificial_training_images_save_path")
    parser.add_argument("--artificial_training_images_box_save_path")

    args = parser.parse_args()
    NUM_TRAINING_IMAGES = args.num_training_images
    BASE_IMAGES_PATH = args.base_images_path
    BASE_IMAGE_BOX_PATH = args.base_images_box_path
    BACKGROUND_IMAGE_PATH = args.background_images_path
    ARTIFICIAL_TRAINING_IMAGES_SAVE_PATH = args.artificial_training_images_save_path
    ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH = args.artificial_training_images_box_save_path

    df = pd.read_csv(BASE_IMAGE_BOX_PATH, delimiter=",",
                     names=["img_path", "x_min", "y_min", "x_max", "y_max", "class_name"])
    df["img_path_short"] = df.img_path.apply(lambda path: os.path.join(BASE_IMAGES_PATH, path[58:]))

    class_list = list(df.class_name)

    base_images_boxes = df[["x_min", "y_min", "x_max", "y_max"]].values

    base_images = [cv2.imread(img_path) for img_path in list(df.img_path_short)]

    background_images_path = [os.path.join(BACKGROUND_IMAGE_PATH, img_name) for img_name in
                              os.listdir(BACKGROUND_IMAGE_PATH)]

    save_train_images_with_labels_and_boxes(num_images_to_generate=NUM_TRAINING_IMAGES,
                                            artificial_training_images_save_path=ARTIFICIAL_TRAINING_IMAGES_SAVE_PATH,
                                            artificial_training_images_box_save_path=ARTIFICIAL_TRAINING_IMAGES_BOX_SAVE_PATH,
                                            class_list=class_list,
                                            base_images_boxes=base_images_boxes,
                                            base_images=base_images,
                                            background_images_path=background_images_path)


if __name__ == '__main__':
    main()
