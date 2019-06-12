import torch
import random
import torchvision.transforms as transforms
import numpy as np
from poissonblending import blend
from PIL import Image
import matplotlib.image as py
import cv2
import os


# 倾斜图片
def process_image(x):
    x_rotate = torch.zeros(x.shape)
    rotate_angle1 = random.uniform(22.5, 45)
    rotate_angle2 = random.uniform(-45, -22.5)
    seed = random.randint(0, 1)  # 随机选择正向还是逆向旋转
    if seed == 0:
        rotate_angle = rotate_angle1
    else:
        rotate_angle = rotate_angle2

    bsize, _, image_h, image_w = x_rotate.shape
    for i in range(bsize):
        image_temp = x[i,:,:,:]
        image_temp = transforms.ToPILImage()(image_temp)
        im2 = image_temp.convert('RGBA') 
        rot = im2.rotate(rotate_angle, expand=1) 
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        image_temp = Image.composite(rot, fff, rot)
        image_temp = image_temp.convert('RGB')
        image_temp = image_temp.crop([image_temp.size[0]/2-128,image_temp.size[0]/2-128,
                    image_temp.size[1]/2+128,image_temp.size[1]/2+128])
        image_temp = transforms.ToTensor()(image_temp)
        x_rotate[i,:,:,:] = image_temp
    return x_rotate


# 进行哈哈镜扭曲（只扭曲，不旋转）
def distort_images(x):
    distort_x = torch.zeros(x.shape)
    bsize, _, image_h, image_w = distort_x.shape
    for i in range(bsize):
        image_temp = x[i,:,:,:]
        image_temp = transforms.ToPILImage()(image_temp)
        image_temp = image_temp.convert('RGBA')
        rand = 'datasets/distorted_images/'+str(random.randint(0, 10000))+'.png'  # 随机生成中间图片的名字
        image_temp.save(rand)  # 保存该张图片
        distort1(rand)  # 对该张图片进行扭曲操作后覆盖原图
        distort_image = Image.open(rand.replace('png', 'jpg'))
        distort_x[i,:,:,:] = transforms.ToTensor()(distort_image)
        os.remove(rand)
        os.remove(rand.replace('png', 'jpg'))
    return distort_x


# 使用另外一种处理方式（扭曲加倾斜）
def distort_images2(x):
    distort_x = torch.zeros(x.shape)
    bsize, _, image_h, image_w = distort_x.shape
    for i in range(bsize):
        image_temp = x[i, :, :, :]
        image_temp = transforms.ToPILImage()(image_temp)
        image_temp = image_temp.convert('RGBA')
        image_temp = image_temp.resize((768, 256), Image.ANTIALIAS)  # 对原始图片尺寸进行调整
        rand = 'datasets/distorted_images/' + str(random.randint(0, 10000)) + '.png'  # 随机生成中间图片的名字
        image_temp.save(rand)  # 保存该张原图片
        distort3(rand)  # 对该张图片进行扭曲操作后保存成jpg
        unrotate(rand.replace('png', 'jpg'))  # 对扭曲旋转后的图片去旋转，只保留扭曲效果
        distort_image = Image.open(rand.replace('png', 'jpg'))  # 读入扭曲后的图片
        distort_image = distort_image.resize((256, 256), Image.ANTIALIAS)
        distort_x[i, :, :, :] = transforms.ToTensor()(distort_image)
        os.remove(rand)
        os.remove(rand.replace('png', 'jpg'))
    return distort_x


# 对一张图中的每个字进行倾斜处理
def distort_images3(x):
    distort_x = torch.zeros(x.shape)
    bsize, _, image_h, image_w = distort_x.shape
    for i in range(bsize):
        image_temp = x[i, :, :, :]
        image_temp = transforms.ToPILImage()(image_temp)
        image_temp = image_temp.convert('RGBA')
        image_temp = image_temp.resize((768, 256), Image.ANTIALIAS)  # 对原始图片尺寸进行调整
        rand = 'datasets/distorted_images/' + str(random.randint(0, 10000)) + '.png'  # 随机生成中间图片的名字
        image_temp.save(rand)  # 保存该张原图片
        distort2(rand)  # 对该张图片进行扭曲操作后保存成jpg
        distort_image = Image.open(rand.replace('png', 'jpg'))  # 读入扭曲后的图片
        distort_image = distort_image.resize((256, 256), Image.ANTIALIAS)
        distort_x[i, :, :, :] = transforms.ToTensor()(distort_image)
        os.remove(rand)
        os.remove(rand.replace('png', 'jpg'))
    return distort_x


# 对一张图片做扭曲处理
# 参数1的取值范围为[0.01,0.03]，参数2的取值范围为[4,8]
def distort1(image_path):
    param1, param2 = random.uniform(0.01, 0.03), random.uniform(4, 7.5)
    img = py.imread(image_path)
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
    n_u, n_v = r.shape[:2]
    diff_u, diff_v = (n_u - u)//2, (n_v - v)//2
    r = r[diff_u:diff_u+u, diff_v:diff_v+v]
    py.imsave(image_path.replace('png', 'jpg'), r)


# 进行每个字倾斜处理，而不是整个图片倾斜处理
def distort2(image_path):
    p1, p2 = random.uniform(0.01, 0.05), random.uniform(4, 7.5)
    p3, p4 = random.uniform(3, 6), random.uniform(0.01, 0.05)

    img = py.imread(image_path)
    u, v = img.shape[:2]

    def f(i, j):
        return i + p1 * np.sin(p2 * np.pi * j)

    def g(i, j):
        return j + p3 * np.sin(p4 * np.pi * i)

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
    r = np.zeros((m1 - n1, m2 - n2, 4))
    for i in range(u):
        for j in range(v):
            i0 = i / u
            j0 = j / v
            u0 = int(f(i0, j0) * u) - n1 - 1
            v0 = int(g(i0, j0) * v) - n2 - 1
            r[u0, v0] = img[i, j]
    py.imsave(image_path.replace('png', 'jpg'), r)


# 进行旋转并拉伸艺术字
def distort3(image_path):
    img = py.imread(image_path)
    u, v = img.shape[:2]

    def f(i, j):
        return i ** 2 - j ** 2

    def g(i, j):
        return 0.8 * i * j

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
    r = np.zeros((m1 - n1, m2 - n2, 4))
    for i in range(u):
        for j in range(v):
            i0 = i / u
            j0 = j / v
            u0 = int(f(i0, j0) * u) - n1 - 1
            v0 = int(g(i0, j0) * v) - n2 - 1
            r[u0, v0] = img[i, j]

    py.imsave(image_path.replace('png', 'jpg'), r)


# 将distort3处理过的图片去旋转
def unrotate(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = - angle - 90

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    cv2.imwrite(image_path, rotated)


def process_image_test(x):
    x_rotate = torch.zeros(x.shape)
    rotate_angle = random.uniform(-45, 45)
    bsize, _, image_h, image_w = x_rotate.shape
    for i in range(bsize):
        image_temp = x[i,:,:,:]
        image_temp = transforms.ToPILImage()(image_temp)
        im2 = image_temp.convert('RGBA') 
        rot = im2.rotate(rotate_angle, expand=1)
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        image_temp = Image.composite(rot, fff, rot)
        image_temp = image_temp.convert('RGB')
        image_temp = image_temp.crop([image_temp.size[0]/2-128,image_temp.size[1]/2-128,
                    image_temp.size[1]/2+128,image_temp.size[1]/2+128])
        image_temp = transforms.ToTensor()(image_temp)
        x_rotate[i,:,:,:] = image_temp
    return x_rotate


def gen_input_mask(
    shape, hole_size,
    hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of distort_output mask.
                A 4D tuple (samples, c, h, w) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 provided,
                holes of size (w, h) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (x, y) of the area,
                while hole_area[1] is its width and height (w, h).
                This area is used as the distort_input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            Input mask tensor with holes.
            All the pixel values within holes are filled with 1,
            while the other pixel values are 0.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    masks = []
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for j in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0
    return mask


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                Size (w, h) of hole area.
        - mask_size (sequence, required)
                Size (w, h) of distort_input mask.
    * returns:
            A sequence which is used for the distort_input argument 'hole_area' of function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A pytorch 4D tensor (samples, c, h, w).
        - area (sequence, required)
                A sequence of length 2 ((x_min, y_min), (w, h)).
                sequence[0] is the left corner of the area to be cropped.
                sequence[1] is its width and height.
    * returns:
            A pytorch tensor cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin : ymin + h, xmin : xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the distort_input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for i in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


# 返回一个随机的测试集batch
def sample_random_test_batch(dataset):
    seed = random.randint(0, len(dataset))
    return dataset[seed]


def poisson_blend(input, output):
    """
    * inputs:
        - distort_input (torch.Tensor, required)
                Input tensor of Completion Network.
        - distort_output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []

    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each distort_input/distort_output image and mask.
    for i in range(num_samples):
        inpainted_np = blend(input_np[i], output_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret
