import numpy as np
from util.data_script import patch_generator,divide_image,merge_image
from cv2 import cv2
# patch_generator("../dataset/CBSD68")
img = cv2.imread("../dataset/CBSD68/3096.png")
patches = divide_image(img,64)
print(patches.dtype)
des_img = merge_image(patches,img,64)
cv2.imshow("ori",img)
cv2.imshow("des",des_img)
print(img.shape,des_img.shape)
cv2.waitKey(0)