import numpy
from pathlib import Path
from cv2 import cv2
import skimage
from tqdm import tqdm
from util.data_script import get_patch

# load images for train
def load_images(input_path,factor):
    p = Path(input_path+"_gt")
    gts,noises = [],[]
    for image_path in tqdm(p.rglob('*.png')):
        if numpy.random.random_integers(1,10) % factor == 0:
            gts.append(cv2.imread(str(image_path)))
            noises.append(cv2.imread(str(image_path).replace("_gt","_noise")))
    return numpy.array(noises),numpy.array(gts)

def patch_generator(files,path):
    Path(path).mkdir(exist_ok=True,parents=True)
    index = 0
    for image_path in tqdm(files):
        img = cv2.imread(image_path)
        imgs = get_patch(img)
        for i,patch in enumerate(imgs):
            cv2.imwrite(path+str(index)+".png",patch)
            index+=1

# generate ssid train data
def extract_train():
    pic_path = "../../dataset/SIDD_Medium_Srgb/Data"
    p = Path(pic_path)
    GT_files,noise_files = [],[]
    for i in p.rglob("*GT*.PNG"):
        GT_files.append(str(i))
        noise_files.append(str(i).replace("GT","NOISY"))
    # patch_generator(GT_files,pic_path+"_gt/")
    # patch_generator(noise_files,pic_path+"_noise/")

# generate ssid test data 
def load_benchmark(input_path):
    pic_path = "../../dataset/SIDD_Benchmark_Data"
    p = Path(pic_path)
    noise_files = []
    for i in p.rglob("*.PNG"):
        noise_files.append(cv2.imread(str(i)))
    return numpy.array(noise_files)