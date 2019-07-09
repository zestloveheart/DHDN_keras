from functools import reduce
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import math
from datetime import datetime
import logging


def list_file(dir, match, scope):
    files = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if scope == "all":
                files.append(os.path.join(dirpath, filename))
            elif scope == "post" and os.path.splitext(filename)[1] == match:
                files.append(os.path.join(dirpath, filename))
    return files


def rename_file(dir, match, scope):
    files = operate_file(dir, "list", match, scope)
    i = 100
    for file in files:
        extension = os.path.splitext(file)[-1]
        dst_name = f"{dir}\\{i}{extension}"
        os.rename(file, dst_name)
        i += 1
    return True


def operate_file(dir, model=None, match=None, scope=None):
    ''' 文件操作
    e.g.
    列出dir中的文件：      operate_file('0')
    列出dir中的.jpg文件：  operate_file('0',match=".jpg",scope="post")
    重命名dir中的文件：    operate_file('0','rename')
    重命名dir中的.jpg文件：operate_file('0','rename',match=".jpg",scope="post")

    :param dir: 需要操作的文件夹路径
    :param model: 选择操作模式："list":列举文件名、"rename":重命名
    :param match: 如果scope为post，则匹配后缀名为 match的文件
    :param scope: 选择操作范围："all":全部文件、"post":按照后缀匹配
    :return:
    '''
    func_dict = {"list": list_file, "rename": rename_file}
    scope_dict = {"all": 1, "post": 2}
    model = model if model in func_dict else "list"
    scope = scope if scope in scope_dict else "all"
    func = func_dict[model]
    return func(dir, match, scope)


def create_data_label_cfg(dir, label, output_file='data_cfg.txt'):
    '''
    读取文件夹dir:str中的图片名，追加到 output_file:str 文件。文件中每一行为：图片名 label:str
    e.g.    create_label('0', '0')     将文件夹0中的图片读取，标注类别为为0
    '''
    filenames = operate_file(dir)
    output_content = f' {label}\n'.join(filenames) + f' {label}\n'
    with open(output_file, 'a') as of:
        of.write(output_content)


def create_classifier_cfg(dir, class_num):
    for i in range(0, class_num):
        temp_path = f"{dir}\\{i}"
        operate_file(temp_path, "rename")
        create_data_label_cfg(temp_path, f"{i}")


def create_classifier_test_cfg(rate=0.1, train_cfg="data_cfg.txt", test_cfg="test_cfg.txt", ):
    # 从数据集中分一部分作为测试集
    with open(train_cfg, 'r', encoding="UTF-8") as file:
        file_content = file.readlines()
    random.shuffle(file_content)

    train_content = file_content[int(rate*len(file_content)):]
    train_str = f''.join(train_content)
    with open(train_cfg, 'w') as file:
        file.write(train_str)

    test_content = file_content[:int(rate * len(file_content))]
    test_str = f''.join(test_content)
    with open(test_cfg, 'w') as file:
        file.write(test_str)


def merge_txt(filenames, output_file="file_merged.txt"):
    # 合并多个txt文件到output_file中
    # 如：merge_txt(["asdf.txt","data_cfg.txt"])
    for file in filenames:
        outstr = ""
        with open(file, 'r', encoding="UTF-8") as file1:
            novel_content = file1.read()
        outstr += novel_content

        with open(output_file, 'a') as file1:
            file1.write(outstr)


def load_data_img(input_file="data_cfg.txt", width=30, height=60, shuffle=False):
    # 根据data_cfg.txt生成data
    data_name, label = [], []
    with open(input_file, 'r') as in_file:
        file_content = in_file.readlines()

    if shuffle:
        random.shuffle(file_content)
    for line in file_content:
        data_name.append(line.split()[0])
        label.append(line.split()[1])
    data = []
    for filename in data_name:
        img = cv2.imread(filename)
        img = cv2.resize(img, (width, height))
        data.append(np.array(img)/255)
    return np.array(data), np.array(label)


def create_yolo_mark_pic_cfg_for_linux():
    dir_name = f"data\\img"  # 图片存放目录
    # rename_file("pic_dig")

    # 把名字变短，主要是为了处理中文。
    operate_file(dir_name, "rename", match=".jpg", scope="post")
    operate_file(dir_name, "rename", match=".txt", scope="post")

    # # 生成train.txt文件
    filenames = operate_file(dir_name, match=".jpg", scope="post")
    print(filenames)
    with open('train_temp.txt', 'w') as f:  # 设置文件对象
        for filename in filenames:
            f.write(filename+"\r")

    # # 把windows的\\斜杠变成linux的/
    with open('train_temp.txt', "r") as f1:  # 设置文件对象
        data = f1.readlines()  # 可以是随便对文件的操作
    filenames = []
    with open('train.txt', 'w') as f2:
        for line in data:
            b = line.replace("\\", "/")
            f2.write(b)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(
        f"{predicted_label} {100*np.max(predictions_array):2.0f}% ({true_label})", color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_result(predictions, test_labels, test_data, num_rows=5, num_cols=3):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_data)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)


def plot_history(histories, key='acc'):
    # plot_history([('baseline', baseline_history),
    #               ('cnn1', cnn_history),
    #               ('lenet', lenet_history),
    #               ('alexnet', alexnet_history)])
    plt.figure(figsize=(16, 10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key],
                       '--', label=name.title() + ' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title() + ' Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def estimate_grade(score: int or float, down_limit: list):
    '''
    表驱动法获取一个项的类别，
    如设定down_limit=[0,60,70,80,90,100]
    score=1，返回0；score=62，返回1；score=83，返回3
    '''
    for i in range(len(down_limit)-1, -1, -1):
        if score > down_limit[i]:

            return i
    return -1


def statistic_count(data: list, opt=0):
    '''
    从list：['a','a','b','b','b','b','c','c','c']的输入
    得到dict：{'a':2,'b':4,'c':3}的输出
    '''
    if opt == 0:
        result = {}
        for i in data:
            if i in result:
                result[i] += 1
            else:
                result[i] = 1
        temp = list(result.items())
        temp.sort()
        return temp
    elif opt == 1:
        result = {}
        for i in data:
            temp = tuple(i)
            if temp in result:
                result[temp] += 1
            else:
                result[temp] = 1
        temp = list(result.items())
        return temp


def generate_x_y_from_list(data: list):
    '''
    从一个[(x1,y1),(x2,y2),(xi,yi),(xn,yn)]的输入
    获取[x1,x2,xi,xn][y1,y2,yi,yn]的输出
    '''
    x, label = [], []
    for i in data:
        x.append(i[0])
        label.append(i[1])
    return x, label


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

    '''
    出自：https://github.com/yu4u/cutout-random-erasing
    使用方法：
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=pixel_level))

    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=callbacks,verbose=1)
    '''


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# import tensorflow as tf
# import numpy as np
# import math
# tf.enable_eager_execution()
# def psnr_metrics(y_true,y_pred):
#     # img1 and img2 have range [0, 255]
#     # img1 = img1.astype(np.float64)
#     # img2 = img2.astype(np.float64)
#     mse = np.mean((y_true - y_pred)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))
    
# y_true = tf.Variable(tf.ones(shape=(64,64,3)))
# y_pred = tf.Variable(tf.zeros(shape=(64,64,3)))
# print(psnr_metrics(y_true,y_pred))
# print(tf.image.psnr(y_true,y_pred,255))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)