from matplotlib import pyplot as plt
from . import nputil

LAYOUT_NP = 'np'
LAYOUT_NHW = 'nhw'
LAYOUT_NHWC = 'nhwc'
LAYOUT_NCHW = 'nchw'

def display_images(images, labels=None, n_cols=8):
    if labels is not None and labels.shape[1]>1:
        labels=nputil.argmax(labels)
    n_rows = -(-len(images) // n_cols)
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()