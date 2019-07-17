import numpy
from pathlib import Path
from cv2 import cv2
import skimage
from tqdm import tqdm


def measure_power(x):
    return float(sum([(float(1) / float(len(x))) * numpy.absolute(x[i]) ** 2 for i in range(0, len(x))]))

def generate_random_vector(signal):
    re = numpy.random.normal(0, 1, len(signal))
    im = numpy.random.normal(0, 1, len(signal))
    return [numpy.complex(re[x], im[x]) for x in range(0, len(re))]

def convert_snr_to_lin(snr):
    return float(10 ** (snr / 20))

def noise_signal(orig_signal, snr):
    variance = numpy.sqrt(measure_power(orig_signal) / (2 * convert_snr_to_lin(snr)))
    random_data = generate_random_vector(orig_signal)
    print (len(random_data), variance)
    noise = [random_data[i] * variance for i in range(0, len(random_data))]
    return [(orig_signal[i]) + noise[i] for i in range(0, len(orig_signal))]

def test_noise():
    import matplotlib.pyplot as plot
    import numpy as np
    x = np.linspace(-2*np.pi, 2*np.pi, 500)
    y = 10*np.sin(x)
    z = noise_signal(y, 10)
    plot.plot(x, y)
    plot.plot(x, z)
    plot.show()

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
        
# load images for train
def load_images(input_path,factor=1):
    p = Path(input_path)
    images = []
    for image_path in tqdm(p.rglob('*.png')):
        if numpy.random.random_integers(1,10) % factor == 0:
            img = cv2.imread(str(image_path))
            images.append(img)
    return numpy.array(images)

# load data for test
def get_data(input_path,var=0.02):
    images = load_images(input_path)
    noise_images = []
    for i in images:
        noise_image = skimage.util.random_noise(i, mode='gaussian',mean=0,var=var)
        noise_image = (noise_image*255).astype(numpy.uint8)
        noise_images.append(noise_image)
    return numpy.array(noise_images),numpy.array(images)

def divide_image(img,size=64):
    origin_shape = img.shape
    des_img = numpy.zeros((size*(origin_shape[0]//size+1),size*(origin_shape[1]//size+1),3),img.dtype)
    des_img[:origin_shape[0],:origin_shape[1],:] = img
    patches = []
    for i in range(0,des_img.shape[0],size):
        for j in range(0,des_img.shape[1],size):
            patches.append(des_img[i:i+size,j:j+size,:])
    return numpy.array(patches)

def merge_image(patches,img,size=64):
    origin_shape = img.shape
    des_img = numpy.zeros((size*(origin_shape[0]//size+1),size*(origin_shape[1]//size+1),3),img.dtype)
    index = 0
    for i in range(0,des_img.shape[0],size):
        for j in range(0,des_img.shape[1],size):
            des_img[i:i+size,j:j+size,:] = patches[index]
            index += 1
    return des_img[:origin_shape[0],:origin_shape[1],:]
            

# data argument for keras 
def add_noise(var=0.02):
    def noising(image):
        return skimage.util.random_noise(image, mode='gaussian',mean=0,var=var)
    return noising


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
