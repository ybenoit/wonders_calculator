import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def draw_rectangles(im, boxes):
    """
    Draw boxes on image

    im: Image to draw boxes on
    boxes: List of boxes (each box must be organized as the following : ["x_0", "y_0", "x_1", "y_1"])
    """

    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Create figure and axes
    fig,ax = plt.subplots(1, figsize=(12,12))
    # Display the image
    ax.imshow(im_rgb)

    for box in boxes :
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()
