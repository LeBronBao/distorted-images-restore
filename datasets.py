import torch.utils.data as data
import os
import torch
import tqdm
import imghdr
import random
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class ImageDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(ImageDataset, self).__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.transform = transform
        self.imgpaths = self.__load_imgpaths_from_dir(self.data_dir)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, index, color_format='RGB'):
        img = Image.open(self.imgpaths[index])
        img = img.convert(color_format)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __is_imgfile(self, filepath):
        filepath = os.path.expanduser(filepath)
        if os.path.isfile(filepath) and imghdr.what(filepath):
            return True
        else:
            return False

    def __load_imgpaths_from_dir(self, dirpath, walk=False, allowed_formats=None):
        imgpaths = []
        dirpath = os.path.expanduser(dirpath)
        if walk:
            for (root, dirs, files) in os.walk(dirpath):
                for file in files:
                    file = os.path.join(root, file)
                    if self.__is_imgfile(file):
                        imgpaths.append(file)
        else:
            for path in os.listdir(dirpath):
                path = os.path.join(dirpath, path)
                if self.__is_imgfile(path) == False:
                    continue
                imgpaths.append(path)
        return imgpaths


class ImagePairsDataset:
    def __init__(self, data_path, batch_size, train_size, image_height, image_width):
        self.data_path = data_path  # 数据集目录
        self.all_pairs = []
        self.train_pairs = []
        self.test_pairs = []
        self.train_batches = []  # 根据batch_size将数据集划分为多个batch
        self.test_batches = []
        self.train_batches_tensors = []  # 保存每个batch中图像的tensor形式
        self.test_batches_tensors = []
        self.batch_size = batch_size
        self.train_size = train_size  # 训练集占全集的比例
        self.image_height = image_height
        self.image_width = image_width

    # 读取图片及其扭曲结果作为数据全集
    def read_image_pairs(self):
        sub_dirs = os.listdir(self.data_path)
        for sub_dir in sub_dirs:
            image_dir_path = os.path.join(self.data_path, sub_dir)
            image_paths = os.listdir(image_dir_path)
            for img_path in image_paths:
                if 'ori' not in img_path:
                    dist_image_abs_path = os.path.join(image_dir_path, img_path)
                    ori_image_abs_path = os.path.join(image_dir_path, 'ori.jpg')
                    self.all_pairs.append((ori_image_abs_path, dist_image_abs_path))

    # 划分训练集和测试集
    def split_train_test(self, shuffle=True):
        if shuffle:
            random.shuffle(self.all_pairs)

        train_num = int(len(self.all_pairs)*self.train_size)
        self.train_pairs = self.all_pairs[:train_num]
        self.test_pairs = self.all_pairs[train_num:]

    # 将训练集、测试集划分为若干个batch
    def split_to_batches(self):
        for i in range(0, len(self.train_pairs), self.batch_size):
            if i+self.batch_size < len(self.train_pairs):
                self.train_batches.append(self.train_pairs[i:i+self.batch_size])
            else:
                self.train_batches.append(self.train_pairs[i:])

        for i in range(0, len(self.test_pairs), self.batch_size):
            if i+self.batch_size < len(self.test_pairs):
                self.test_batches.append(self.test_pairs[i:i+self.batch_size])
            else:
                self.test_batches.append(self.test_pairs[i:])

    # 将训练集和测试集图片读入并转为Tensor
    def trans_images_to_tensors(self):
        i = 0
        for batch in self.train_batches:
            real_tensors = []
            dist_tensors = []
            for pair in batch:
                path1, path2 = pair[0], pair[1]
                img1 = Image.open(path1)
                img2 = Image.open(path2)
                img1_arr = np.array(img1)  # 转为nd-array
                img2_arr = np.array(img2)
                img1_arr = img1_arr.reshape([3, self.image_height, self.image_width])  # 调整维度
                img2_arr = img2_arr.reshape([3, self.image_height, self.image_width])
                real_tensors.append(img1_arr)
                dist_tensors.append(img2_arr)
            real_tensors = torch.Tensor(real_tensors)  # 转为tensor
            dist_tensors = torch.Tensor(dist_tensors)
            self.train_batches_tensors.append((real_tensors, dist_tensors))
            print('Finished tensorizing training batch:'+str(i))
            i += 1

        i = 0
        for batch in self.test_batches:  # 对于测试集只需要将扭曲后的图片转为tensor
            dist_tensors = []
            for pair in batch:
                path2 = pair[1]
                img2 = Image.open(path2)
                # 转为nd-array
                img2_arr = np.array(img2)
                # 调整维度
                img2_arr = img2_arr.reshape([3, self.image_height, self.image_width])
                dist_tensors.append(img2_arr)
            # 转为tensor
            dist_tensors = torch.Tensor(dist_tensors)
            self.test_batches_tensors.append(dist_tensors)
            print('Finished tensorizing testing batch:' + str(i))
            i += 1

    def prepare_tensors(self):
        self.read_image_pairs()
        self.split_train_test()
        self.split_to_batches()
        self.trans_images_to_tensors()

    def get_train_test_tensor(self):
        self.prepare_tensors()
        return self.train_batches_tensors, self.test_batches_tensors


def distort_test(image_path, param1, param2):
    import numpy as np
    import matplotlib.image as py
    img2 = Image.open(image_path)
    img = np.array(img2)
    #img = py.imread(image_path)
    u, v = img.shape[:2]

    def f(i, j):
        return i + param1 * np.sin(param2 * np.pi * j)

    def g(i, j):
        return j + param1 * np.sin(param2 * np.pi * i)

    M = []
    N = []
    for i in range(u):
        for j in range(v):
            i0 = i / u
            j0 = j / v
            u0 = int(f(i0, j0) * u)
            v0 = int(g(i0, j0) * v)
            M.append(u0)
            N.append(v0)
    m1, m2 = max(M), max(N)
    n1, n2 = min(M), min(N)
    r = np.zeros((m1-n1, m2-n2, 4))
    for i in range(u):
        for j in range(v):
            i0 = i / u
            j0 = j / v
            u0 = int(f(i0, j0) * u)-n1-1
            v0 = int(g(i0, j0) * v)-n2-1
            r[u0, v0] = img[i, j]



if __name__ == '__main__':
    print()