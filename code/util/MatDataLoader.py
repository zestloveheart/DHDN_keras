import numpy as np
from pathlib import Path
from cv2 import cv2
import h5py
import skimage
import scipy.io as scio
import imageio
import time
import random
class MatDataLoader(object):
    def __init__(self, GT_path, batch_size, target_size=256,noise_path=None, noise_coefficient=0.2, seed=5, *args, **kwargs):
        self.GT_path = GT_path
        self.batch_size = batch_size
        self.GT_p = Path(self.GT_path)
        self.file_indexes = [str(i) for i in self.GT_p.rglob('*GT*.MAT')]
        self.target_size = target_size
        self.seed = seed

    def __len__(self):
        return len(self.file_indexes)

    # get images for test
    def get_images(self,indexes):
        GT_images, noise_images = [],[]
        for file_index in indexes:
            gt = h5py.File(file_index)['x']
            noise = h5py.File(file_index.replace('GT','NOISY'))['x']
            GT_images.append(gt)
            noise_images.append(noise)
            # images.append(data['x'][0:500,0:500])
        return GT_images,noise_images

    # load images for train
    def load_images_train(self):
        np.random.seed(self.seed)
        while True:
            i = 0
            while i + self.batch_size < self.__len__():
                GT_indexes = self.file_indexes[i:i+self.batch_size]
                GT_batch, noise_batch = self.get_images(GT_indexes)
                for index in range(self.batch_size):
                    pattern = self.classify_pattern(GT_indexes[index])
                    random_recorder = random.randint(0,3)
                    # get patch to reduce computation
                    GT_batch[index] = self.get_patch(GT_batch[index],random_recorder,20)
                    noise_batch[index] = self.get_patch(noise_batch[index],random_recorder,20)
                    # bayer unify
                    GT_batch[index] = self.bayer_unify_crop(GT_batch[index],pattern)
                    noise_batch[index] = self.bayer_unify_crop(noise_batch[index],pattern)
                    # bayer aug
                    GT_batch[index] = self.bayer_aug(GT_batch[index],random_recorder)
                    noise_batch[index] = self.bayer_aug(noise_batch[index],random_recorder)
                    
                    # get patch for fixed 64 
                    GT_batch[index] = self.get_patch(GT_batch[index],random_recorder)
                    noise_batch[index] = self.get_patch(noise_batch[index],random_recorder)

                    GT_batch[index] = self.pack_raw(GT_batch[index])
                    noise_batch[index] = self.pack_raw(noise_batch[index])
                yield np.array(noise_batch), np.array(GT_batch)
                i += self.batch_size
    
    # get n * img(64x64) from a whole img
    def get_patch(self,img,seed,big=0):
        np.random.seed(seed)
        start_h = 2*np.random.randint(0,(img.shape[0]-self.target_size-big)//2)
        start_w = 2*np.random.randint(0,(img.shape[1]-self.target_size-big)//2)
        return img[start_h:start_h+self.target_size+big,start_w:start_w+self.target_size+big]

    def show_mat(self,raw,title="raw"):
        data = (raw*255).astype(np.uint8)
        # test different model
        # li = [cv2.COLOR_BayerBG2BGR, cv2.COLOR_BayerGB2BGR, cv2.COLOR_BayerRG2BGR, cv2.COLOR_BayerGR2BGR, cv2.COLOR_BayerBG2RGB, cv2.COLOR_BayerGB2RGB, cv2.COLOR_BayerRG2RGB, cv2.COLOR_BayerGR2RGB]
        # for i in li:
        #     data_temp = cv2.cvtColor(data1, i, data_temp)
        #     cv2.imshow("asdf",data_temp)
        #     cv2.waitKey(0)
        data_temp = cv2.cvtColor(data, cv2.COLOR_BayerBG2RGB)
        cv2.imshow(title,data_temp)
        cv2.waitKey(0)

    def classify_pattern(self,filename):
        device_pattern = {'IP':'rggb','S6':'grbg'}
        for i in device_pattern.keys():
            if i in filename:
                return device_pattern[i]

    def bayer_unify_crop(self,img,pattern=None):
        # origin to BGGR for train
        # BGGR to origin for test as disunify
        if pattern == 'grbg':
            return img[1:-1,:]
        elif pattern == 'gbrg':
            return img[:,1:-1]
        elif pattern == 'rggb':
            return img[1:-1,1:-1]
        else:
            return img

    def bayer_unify_pad(self,img,pattern=None):
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        # origin to BGGR for test
        if pattern == 'grbg':
            return np.pad(img, ((1, 1), (0, 0)), 'constant', constant_values=0)
        elif pattern == 'gbrg':
            return np.pad(img, ((0, 0), (1, 1)), 'constant', constant_values=0)
        elif pattern == 'rggb':
            return np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=0)
        else:
            return img

    def bayer_aug(self,img,mode):
        mode_dict = {0:'trans',1:'hor',2:'ver',3:''}
        mode = mode_dict[mode]
        if mode == 'hor':
            return cv2.flip(img,1)[:,1:-1]
        elif mode == 'ver':
            return img[::-1][1:-1,:]
        elif mode == 'trans':
            return img.T
        return img
        
    def pack_raw(self,raw):
        raw_shape = raw.shape
        H = raw_shape[0]
        W = raw_shape[1]
        out = np.concatenate([raw[0:H:2, 0:W:2].reshape((H//2,W//2,1)),
                            raw[0:H:2, 1:W:2].reshape((H//2,W//2,1)),
                            raw[1:H:2, 1:W:2].reshape((H//2,W//2,1)),
                            raw[1:H:2, 0:W:2].reshape((H//2,W//2,1))],axis=2)
        return out

if __name__ == "__main__":
    mdl = MatDataLoader('.',4,128)
    print(mdl.__len__())
    for i,j in mdl.load_images_train():
        # print(i.shape,j.shape)
        # time.sleep(2)
        for ii,jj in zip(i,j):
            print(ii.shape)
            time.sleep(1)
            # mdl.show_mat(ii,"GT")
            # mdl.show_mat(jj,"NOISY")
        