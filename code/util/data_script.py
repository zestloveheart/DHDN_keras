import numpy
from pathlib import Path
from cv2 import cv2
import skimage

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

def get_patch(img,size=64):
    patchs = []
    
    for i in range(0,img.shape[0],size):
        for j in range(0,img.shape[1],size):
            if i+size < img.shape[0] and j+size < img.shape[1]:
                patchs.append(img[i:i+size,j:j+size,:])
    return patchs

def load_images(input_path):
    p = Path(input_path)
    images = []
    for image_path in p.rglob('*.png'):
        img = cv2.imread(str(image_path))
        img = get_patch(img)
        images += img
    return numpy.array(images)

def get_data(input_path,var=0.02):
    images = load_images(input_path)
    noise_image = skimage.util.random_noise(images, mode='gaussian',mean=0,var=var)
    noise_image = (noise_image*255).astype(numpy.uint8)
    return noise_image,images

def add_noise(var=0.02):
    def noising(image):
        return skimage.util.random_noise(image, mode='gaussian',mean=0,var=var)
    return noising
