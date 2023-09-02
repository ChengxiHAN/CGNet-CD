import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance


def cv_random_flip(img_A, img_B, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label


def randomCrop_Mosaic(image_A, image_B, label, crop_win_width, crop_win_height):
    image_width = image_A.size[0]
    image_height = image_A.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)


def randomCrop(image_A, image_B, label):
    border = 30
    image_width = image_A.size[0]
    image_height = image_B.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image_A.crop(random_region), image_B.crop(random_region), label.crop(random_region)


def randomRotation(image_A, image_B, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image_A = image_A.rotate(random_angle, mode)
        image_B = image_B.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    bright_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Brightness(image_A).enhance(bright_intensity)
    image_B = ImageEnhance.Brightness(image_B).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Contrast(image_A).enhance(contrast_intensity)
    image_B = ImageEnhance.Contrast(image_B).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image_A = ImageEnhance.Color(image_A).enhance(color_intensity)
    image_B = ImageEnhance.Color(image_B).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image_A = ImageEnhance.Sharpness(image_A).enhance(sharp_intensity)
    image_B = ImageEnhance.Sharpness(image_B).enhance(sharp_intensity)
    return image_A, image_B


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
class ChangeDataset(data.Dataset):
    def __init__(self, root, trainsize, mosaic_ratio=0.75):
        self.trainsize = trainsize
        # get filenames
        self.image_root_A =  root + 'A/'
        self.image_root_B =  root + 'B/'
        self.gt_root = root + 'label/'
        self.mosaic_ratio = mosaic_ratio
        self.images_A = [self.image_root_A + f for f in os.listdir(self.image_root_A) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_B = [self.image_root_B + f for f in os.listdir(self.image_root_B) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
        self.gts = sorted(self.gts)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images_A)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio:
            image_A, image_B, gt = self.load_img_and_mask(index)
            image_A, image_B, gt = cv_random_flip(image_A, image_B, gt)
            image_A, image_B, gt = randomCrop(image_A, image_B, gt)
            image_A, image_B, gt = randomRotation(image_A, image_B, gt)
            image_A,image_B = colorEnhance(image_A,image_B)
            gt = randomPeper(gt)
        else:
            image_A, image_B, gt = self.load_mosaic_img_and_mask(index)
            image_A, image_B, gt = cv_random_flip(image_A, image_B, gt)
            image_A, image_B, gt = randomRotation(image_A, image_B, gt)
            image_A,image_B = colorEnhance(image_A,image_B)
            gt = randomPeper(gt)

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)

        return image_A, image_B, gt


    def load_img_and_mask(self, index):
        A = Image.open(self.images_A[index]).convert('RGB')
        B = Image.open(self.images_B[index]).convert('RGB')
        mask = Image.open(self.gts[index]).convert('L')
        return A, B, mask

    def load_mosaic_img_and_mask(self, index):
       indexes = [index] + [random.randint(0, self.size - 1) for _ in range(3)]
       img_a_a, img_a_b, mask_a = self.load_img_and_mask(indexes[0])
       img_b_a, img_b_b, mask_b = self.load_img_and_mask(indexes[1])
       img_c_a, img_c_b, mask_c = self.load_img_and_mask(indexes[2])
       img_d_a, img_d_b, mask_d = self.load_img_and_mask(indexes[3])

       w = self.trainsize
       h = self.trainsize

       start_x = w // 4
       strat_y = h // 4
        # The coordinates of the splice center
       offset_x = random.randint(start_x, (w - start_x))
       offset_y = random.randint(strat_y, (h - strat_y))

       crop_size_a = (offset_x, offset_y)
       crop_size_b = (w - offset_x, offset_y)
       crop_size_c = (offset_x, h - offset_y)
       crop_size_d = (w - offset_x, h - offset_y)

       croped_a_a, croped_a_b, mask_crop_a = randomCrop_Mosaic(img_a_a.copy(), img_a_b.copy(), mask_a.copy(),crop_size_a[0], crop_size_a[1]) 
       croped_b_a, croped_b_b, mask_crop_b = randomCrop_Mosaic(img_b_a.copy(), img_b_b.copy(), mask_b.copy(),crop_size_b[0], crop_size_b[1])
       croped_c_a, croped_c_b, mask_crop_c = randomCrop_Mosaic(img_c_a.copy(), img_c_b.copy(), mask_c.copy(),crop_size_c[0], crop_size_c[1])
       croped_d_a, croped_d_b, mask_crop_d = randomCrop_Mosaic(img_d_a.copy(), img_d_b.copy(), mask_d.copy(),crop_size_d[0], crop_size_d[1])

       croped_a_a, croped_a_b, mask_crop_a = np.array(croped_a_a), np.array(croped_a_b), np.array(mask_crop_a)
       croped_b_a, croped_b_b, mask_crop_b = np.array(croped_b_a), np.array(croped_b_b), np.array(mask_crop_b)
       croped_c_a, croped_c_b, mask_crop_c = np.array(croped_c_a), np.array(croped_c_b), np.array(mask_crop_c)
       croped_d_a, croped_d_b, mask_crop_d = np.array(croped_d_a), np.array(croped_d_b), np.array(mask_crop_d)

       top_a = np.concatenate((croped_a_a, croped_b_a), axis=1)
       bottom_a = np.concatenate((croped_c_a, croped_d_a), axis=1)
       img_a = np.concatenate((top_a, bottom_a), axis=0)

       top_b = np.concatenate((croped_a_b, croped_b_b), axis=1)
       bottom_b = np.concatenate((croped_c_b, croped_d_b), axis=1)
       img_b = np.concatenate((top_b, bottom_b), axis=0)

       top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
       bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
       mask = np.concatenate((top_mask, bottom_mask), axis=0)
       mask = np.ascontiguousarray(mask)


       img_a = np.ascontiguousarray(img_a)
       img_b = np.ascontiguousarray(img_b)

       img_a = Image.fromarray(img_a)
       img_b = Image.fromarray(img_b)
       mask = Image.fromarray(mask)

       return img_a, img_b, mask

    def filter_files(self):
        assert len(self.images_A) == len(self.gts)
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        gts = []
        edges = []
        for img_A_path, img_B_path, gt_path in zip(self.images_A, self.images_B, self.gts):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            gt = Image.open(gt_path)
            if img_A.size == img_B.size:
                if img_A.size == gt.size:
                    images_A.append(img_A_path)
                    images_B.append(img_B_path)
                    gts.append(gt_path)

        self.images_A = images_A
        self.images_B = images_B
        self.gts = gts


    def __len__(self):
        return self.size

def get_loader(root, batchsize, trainsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =ChangeDataset(root = root, trainsize= trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class Test_ChangeDataset(data.Dataset):
    def __init__(self, root, trainsize):
        self.trainsize = trainsize
        # get filenames
        image_root_A =  root + 'A/'
        image_root_B =  root + 'B/'
        gt_root = root + 'label/'
        self.images_A = [image_root_A + f for f in os.listdir(image_root_A) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_B = [image_root_B + f for f in os.listdir(image_root_B) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
        self.gts = sorted(self.gts)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images_A)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image_A = self.rgb_loader(self.images_A[index])
        image_B = self.rgb_loader(self.images_B[index])
        gt = self.binary_loader(self.gts[index])
        # data augumentation

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        gt = self.gt_transform(gt)
        file_name = self.images_A[index].split('/')[-1][:-len(".png")]

        return image_A, image_B, gt, file_name

    def filter_files(self):
        assert len(self.images_A) == len(self.gts)
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        gts = []
        for img_A_path, img_B_path, gt_path in zip(self.images_A, self.images_B, self.gts):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            gt = Image.open(gt_path)
            if img_A.size == img_B.size:
                if img_A.size == gt.size:
                    images_A.append(img_A_path)
                    images_B.append(img_B_path)
                    gts.append(gt_path)

        self.images_A = images_A
        self.images_B = images_B
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

def get_test_loader(root, batchsize, trainsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =Test_ChangeDataset(root = root, trainsize= trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


