import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms
import numpy as np


# -------------------------------
# data is orgnized as follows:
# --data_roots
# ----imageX_dir
# ------imageX_1
# ------imageX_2
# ------ ......
# ------imageX_m
# ----imageY_dir
# ------imageY_1
# ------imageY_2
# ------ ......
# ------imageY_m
# -------------------------------

# dataset class
class dataset(data.Dataset):
    def __init__(self, data_roots, scale_factor):
        super(dataset, self).__init__()
        self.data_roots = data_roots
        self.scale_factor = scale_factor
        self.imageX_name_list, self.imageY_name_list = self.get_image_name_list()

    def __getitem__(self, index):
        imageX = load_image(self.imageX_name_list[index])
        imageZ = load_image(self.imageY_name_list[index])

        # imageX = np.abs(np.array(imageX) - 94.5137) / 40.3980
        # imageZ = np.abs(np.array(imageZ) - 107.2594) / 35.1247
        #
        # imageX = 2 * (np.array(imageX) / 255.0 - 0.5)
        # imageZ = 2 * (np.array(imageZ) / 255.0 - 0.5)

        imageX = np.array(imageX)
        imageZ = np.array(imageZ)

        imageX = Image.fromarray(imageX)
        imageZ = Image.fromarray(imageZ)

        imageX = down_scale(imageX, self.scale_factor)  # NLR image
        imageY = down_scale(imageZ, self.scale_factor)  # CLR image
        imageZ = to_tensor(imageZ)  # CHR image
        return imageX, imageY, imageZ

    def get_image_name_list(self):
        for name in os.listdir(self.data_roots):
            if 'noisy' in name:
                dir_path = os.path.join(self.data_roots, name)
                imageX_name_list = [os.path.join(dir_path, imageX) for imageX in os.listdir(dir_path) if
                                    is_image_file(imageX)]
            if 'clear' in name:
                dir_path = os.path.join(self.data_roots, name)
                imageY_name_list = [os.path.join(dir_path, imageY) for imageY in os.listdir(dir_path) if
                                    is_image_file(imageY)]
        return imageX_name_list, imageY_name_list

    def __len__(self):
        return len(self.imageX_name_list)


# basic functions
def is_image_file(file_name):
    return any(file_name.endswith(extension) for extension in ['.jpg', '.png', '.jpeg', '.tif'])


def load_image(file_name):
    return Image.open(file_name).convert('L')


def down_scale(image, scale_factor):
    h, w = image.size
    # down_scale = transforms.Compose([transforms.Resize((h // scale_factor, w // scale_factor), Image.BICUBIC),
    # transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    down_scale = transforms.Compose(
        [transforms.Resize((h // scale_factor, w // scale_factor), Image.BICUBIC), transforms.ToTensor()])
    return down_scale(image)


def to_tensor(image):
    # return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])(image)
    return transforms.ToTensor()(image)




