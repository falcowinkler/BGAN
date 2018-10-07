import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def save_image(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.system("java -jar " + os.getcwd() + "/lib/copicat.jar -i " + input_dir + " -o " + output_dir \
              + " --input-format protobuf --output-format image")
    return output_dir


def display_image(path_to_file):
    img = mpimg.imread(path_to_file)
    imgplot = plt.imshow(img)
    plt.show()
