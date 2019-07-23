import numpy as np
from pathlib import Path
from cv2 import cv2
from keras.preprocessing.image import ImageDataGenerator
import skimage
import time
from tqdm import tqdm

class RgbDataLoader(object):
    def __init__(self, GT_path, batch_size, target_size=64,noise_path=None, noise_coefficient=0.02, seed=5, *args, **kwargs):
        self.GT_path = GT_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.noise_path = noise_path
        if not self.noise_path:
            self.noise_coefficient = noise_coefficient
        self.seed = seed

        self.GT_p = Path(self.GT_path)
        self.data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                    # validation_split=0.1
                     )
        if not Path(self.GT_path+"_patch").exists():
            self.patch_generator(self.GT_path)
        self.GT_p = Path(self.GT_path+"_patch")
        self.file_indexes = [str(i) for i in self.GT_p.rglob('*.png')]

    def __len__(self):
        return len(self.file_indexes)

    # get n * img(64x64) from a whole img
    def get_patch(self,img,size=64):
        patchs = []
        for i in range(0,img.shape[0],size):
            for j in range(0,img.shape[1],size):
                if i+size < img.shape[0] and j+size < img.shape[1]:
                    patchs.append(img[i:i+size,j:j+size,:])
        return patchs

    # generate 64x64 patch image for train
    def patch_generator(self,input_path):
        p = Path(input_path)
        output_path = input_path+"_patch/GT/"
        Path(output_path).mkdir(exist_ok=True,parents=True)
        index = 0
        for image_path in tqdm(p.rglob('*.png')):
            img = cv2.imread(str(image_path))
            imgs = self.get_patch(img)
            for i,patch in enumerate(imgs):
                cv2.imwrite(output_path+str(index)+".png",patch)
                index+=1

    # data argument noise for keras 
    def add_noise(self,var=0.02):
        def noising(image):
            return skimage.util.random_noise(image/255, mode='gaussian',mean=0,var=var)
        return noising

    # load images for train
    def load_images_train(self):
        if not self.noise_path:
            noise_datagen = ImageDataGenerator(preprocessing_function=self.add_noise(var=self.noise_coefficient),**self.data_gen_args)
            GT_datagen = ImageDataGenerator(preprocessing_function=self.add_noise(var=0.),**self.data_gen_args)
            noise_generator = noise_datagen.flow_from_directory(self.GT_p,batch_size=self.batch_size,target_size=(self.target_size,self.target_size),class_mode=None ,seed=self.seed)
            GT_generator = GT_datagen.flow_from_directory(self.GT_p,batch_size=self.batch_size,target_size=(self.target_size,self.target_size),class_mode=None,seed=self.seed)
            return zip(noise_generator, GT_generator)

    def show_mat(self,raw,title="raw"):
        cv2.imshow(title,raw)
        cv2.waitKey(0)

if __name__ == "__main__":
    mdl = RgbDataLoader('../dataset/CBSD68',8,target_size=64)
    for i,j in mdl.load_images_train():
        # time.sleep(2)
        for ii,jj in zip(i,j):
            # print(ii[:10,:10,0])
            # time.sleep(2)
            mdl.show_mat(ii,"GT")
            mdl.show_mat(jj,"NOISY")