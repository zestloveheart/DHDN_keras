import numpy as np
from pathlib import Path
from cv2 import cv2
import h5py
import skimage
from keras.preprocessing.image import ImageDataGenerator
import scipy.io as scio
import imageio
import time
class MatDataLoader(object):
    def __init__(self, GT_path, batch_size, target_size=256,noise_path=None, noise_coefficient=0.2, seed=5, *args, **kwargs):
        self.GT_path = GT_path
        self.batch_size = batch_size
        self.GT_p = Path(self.GT_path)
        self.file_indexes = [str(i) for i in self.GT_p.rglob('*GT*.MAT')]
        self.target_size = target_size

    def __len__(self):
        return len(self.file_indexes)

    # get images for test
    def get_images(self,indexes):
        GT_images, noise_images = [],[]
        for file_index in indexes:
            gt = h5py.File(file_index)['x']
            noise = h5py.File(file_index.replace('GT','NOISY'))['x']
            start_h = 2*np.random.randint(0,(gt.shape[0]-self.target_size)//2)
            start_w = 2*np.random.randint(0,(gt.shape[1]-self.target_size)//2)
            GT_images.append(gt[start_h:start_h+self.target_size,start_w:start_w+self.target_size])
            noise_images.append(noise[start_h:start_h+self.target_size,start_w:start_w+self.target_size])
            # images.append(data['x'][0:500,0:500])
        return GT_images,noise_images

    # load images for train
    def load_images_train(self):
        while True:
            i = 0
            while i + self.batch_size < self.__len__():
                GT_indexes = self.file_indexes[i:i+self.batch_size]
                GT_batch, noise_batch = self.get_images(GT_indexes)
                for index in range(self.batch_size):
                    # get patch
                    
                    # bayer unify
                    pattern = self.classify_pattern(GT_indexes[index])
                    GT_batch[index] = self.bayer_unify_crop(GT_batch[index],pattern)
                    noise_batch[index] = self.bayer_unify_crop(noise_batch[index],pattern)
                    # bayer aug
                    random_recorder = np.random.randint(0,3)
                    GT_batch[index] = self.bayer_aug(GT_batch[index],random_recorder)
                    noise_batch[index] = self.bayer_aug(noise_batch[index],random_recorder)
                    
                    GT_batch[index] = self.pack_raw(GT_batch[index])
                    noise_batch[index] = self.pack_raw(noise_batch[index])
                yield noise_batch, GT_batch
                i += self.batch_size

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
        mode_dict = {0:'trans',1:'hor',2:'ver'}
        mode = mode_dict[mode]
        if mode == 'hor':
            return cv2.flip(img,1)[:,1:-1]
        elif mode == 'ver':
            return img[::-1][1:-1,:]
        elif mode == 'trans':
            return img.T
        return img
        
    def pack_raw(self,raw):
        # pack Bayer image to 4 channels
        # im = raw.raw_image_visible.astype(np.float32)
        # im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

        raw_shape = raw.shape
        H = raw_shape[0]
        W = raw_shape[1]

        out = np.concatenate((raw[0:H:2, 0:W:2],
                            raw[0:H:2, 1:W:2],
                            raw[1:H:2, 1:W:2],
                            raw[1:H:2, 0:W:2]), axis=2)
        return out

if __name__ == "__main__":
    mdl = MatDataLoader('.',8,)
    print(mdl.__len__())
    for i,j in mdl.load_images_train():
        # print(i.shape,j.shape)
        # time.sleep(2)
        for ii,jj in zip(i,j):
            # print(ii.shape)
            # time.sleep(2)
            mdl.show_mat(ii,"GT")
            mdl.show_mat(jj,"NOISY")
        



        
# # data augmentation
# # we create two instances with the same arguments

# # Provide the same seed and keyword arguments to the fit and flow methods
# # (std, mean, and principal components if ZCA whitening is applied).
# seed = 5
# # image_datagen.fit(x_train, augment=True, seed=seed)
# # GT_datagen.fit(y_train, augment=True, seed=seed)

# # combine generators into one which yields image and GT
# train_generator = zip(image_generator, GT_generator)




# # load images for train
# def load_images(self):
#     batch_image = []
#     i = 0
#     while True:
#         while i + self.batch_size < self.__len__():
#             temp = numpy.array([cv2.imread(filename)/255 for filename in file_indexes[i:i+self.batch_size]])
#             image_generator = image_datagen.flow(temp,batch_size=batch_size,seed=seed)
#             GT_generator = GT_datagen.flow(temp,batch_size=batch_size,seed=seed)
#             yield 
#             i += self.batch_size
#         else:
#             i = 0

# get n * img(64x64) from a whole img
def get_patch(img,size=64):
    patchs = []
    for i in range(0,img.shape[0],size):
        for j in range(0,img.shape[1],size):
            if i+size < img.shape[0] and j+size < img.shape[1]:
                patchs.append(img[i:i+size,j:j+size,:])
    return patchs

# generate 64x64 patch image for train
def patch_generator(input_path):
    p = Path(input_path)
    output_path = input_path+"_patch/"
    Path(output_path).mkdir(exist_ok=True,parents=True)
    index = 0
    for image_path in tqdm(p.rglob('*.png')):
        img = cv2.imread(str(image_path))
        imgs = get_patch(img)
        for i,patch in enumerate(imgs):
            cv2.imwrite(output_path+str(index)+".png",patch)
            index+=1
        


def test_generator():
    from util import calculate_psnr
    from keras.preprocessing.image import ImageDataGenerator
    import os
    from pathlib import Path
    from cv2 import cv2
    # training configuration
    batch_size = 8

    # set data_path
    # windows
    data_path = 'E:\\zlh\\workspace\\python\\pro3_denoising\\DHDN_keras\\dataset\\CBSD68'
    # 225
    # data_path = '/media/tclwh2/wangshupeng/zoulihua/DHDN_keras/dataset/CBSD68'
    # 11
    # data_path = '/home/tcl/zoulihua/DHDN_keras/dataset/CBSD68'

    # The data, shuffled and split between train and test sets:
    y_train = load_images(data_path) # get_data(data_path,0.01)

    # data augmentation
    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True,
                        vertical_flip=True,
                        #  validation_split=0.1
                        )
    image_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=0.02),**data_gen_args)
    mask_datagen = ImageDataGenerator(preprocessing_function=add_noise(var=0.01),**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    # (std, mean, and principal components if ZCA whitening is applied).
    seed = 5
    # image_datagen.fit(x_train, augment=True, seed=seed)
    # mask_datagen.fit(y_train, augment=True, seed=seed)

    y_train = y_train/255
    image_generator = image_datagen.flow(y_train,batch_size=batch_size,seed=seed)
    mask_generator = mask_datagen.flow(y_train,batch_size=batch_size,seed=seed)
    import numpy as np
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    print(train_generator.__next__()[0].dtype)
    for batch in train_generator:
        x,y = batch
        for index in range(batch_size):
            print(np.mean(x[index]))
            print(np.mean(y[index]))
            cv2.imshow("noise",x[index])
            cv2.imshow("GT",y[index])
            cv2.waitKey(0)
