import os
import requests
import shutil
import scipy.io as sio
import cv2
import skimage.io
import random
from xml.dom import minidom
from tensorflow import keras

from utils import logger, ROOT_DIR


DATA_DIR = os.path.join(ROOT_DIR, 'dog_breeds_data')

IMAGE_DIM = 400


# https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
def download_file(url, directory, filename):
    logger.info(f'Downloading {url}...')
    with requests.get(url, stream=True) as r:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    logger.info(f'File downloaded and saved in {filepath}')


def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs('./dog_breeds_data')
    if not os.path.exists(os.path.join(DATA_DIR, 'images.tar')):
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar', DATA_DIR, 'images.tar')
    if not os.path.exists(os.path.join(DATA_DIR, 'annotation.tar')):
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar', DATA_DIR, 'annotation.tar')
    if not os.path.exists(os.path.join(DATA_DIR, 'lists.tar')):
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar', DATA_DIR, 'lists.tar')


def extract_data():
    extract_file(DATA_DIR, 'images.tar')
    extract_file(DATA_DIR, 'annotation.tar')
    extract_file(DATA_DIR, 'lists.tar', extract_path=DATA_DIR + '/Lists')


def extract_file(directory, filename, extract_path=DATA_DIR):
    logger.info(f'Extracting {filename} into {extract_path}...')
    shutil.unpack_archive(os.path.join(directory, filename), extract_path)
    logger.info(f'{filename} extracted.')
    logger.info(f'Deleting {filename}...')
    os.remove(os.path.join(directory, filename))
    logger.info(f'{filename} deleted.')


def prefix_with_data_folder(path):
    return os.path.join(DATA_DIR, path)


def ensure_existing_data_folder(path):
    folder_path = prefix_with_data_folder(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_crop_size(width, height):
    if width > height:
        scale = IMAGE_DIM/width
    else:
        scale = IMAGE_DIM/height
    new_width, new_height = width*scale, height*scale
    return new_width, new_height


def get_padding_size(width, height):
    top = int((IMAGE_DIM - height) / 2)
    bottom = int((IMAGE_DIM - height) / 2)
    left = int((IMAGE_DIM - width) / 2)
    right = int((IMAGE_DIM - width) / 2)
    return top, bottom, left, right


def modify_image_size(annotation_path, src):
    img = skimage.io.imread(src)
    dom = minidom.parse(annotation_path)
    object_tag = dom.getElementsByTagName('object')
    bndbox_tag = object_tag[0].getElementsByTagName('bndbox')
    xmin = int(bndbox_tag[0].getElementsByTagName('xmin')[0].firstChild.nodeValue)
    ymin = int(bndbox_tag[0].getElementsByTagName('ymin')[0].firstChild.nodeValue)
    xmax = int(bndbox_tag[0].getElementsByTagName('xmax')[0].firstChild.nodeValue)
    ymax = int(bndbox_tag[0].getElementsByTagName('ymax')[0].firstChild.nodeValue)
    cropped_img = img[ymin:ymax, xmin:xmax]

    width = xmax - xmin
    height = ymax - ymin
    if width > IMAGE_DIM or height > IMAGE_DIM:
        width, height = get_crop_size(width, height)
        cropped_img = cv2.resize(cropped_img, (int(width), int(height)))

    border_type = cv2.BORDER_CONSTANT
    top, bottom, left, right = get_padding_size(width, height)
    cropped_img = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, border_type)
    return cropped_img


def create_data_subset(image_list, destination_folder):
    logger.info(f'Creating {destination_folder} subset...')
    images_folder = prefix_with_data_folder('Images')
    annotations_folder = prefix_with_data_folder('Annotation')
    ensure_existing_data_folder(destination_folder)
    for image_path in image_list:
        src = os.path.join(images_folder, image_path)
        dst = os.path.join(prefix_with_data_folder(destination_folder), image_path)
        class_folder = image_path.split('/')[0]
        ensure_existing_data_folder(os.path.join(destination_folder, class_folder))
        annotation_str = image_path.split('.')[0]
        annotation_path = os.path.join(annotations_folder, annotation_str)
        cropped_img = modify_image_size(annotation_path, src)
        cv2.imwrite(dst, cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))


def get_split_data_lists():
    images_folder = prefix_with_data_folder('Images')
    class_list = os.listdir(images_folder)
    train_list = []
    test_list = []

    for class_path in class_list:
        class_folder = os.path.join(images_folder, class_path)
        class_images_list = os.listdir(class_folder)
        random.seed(0)
        random.shuffle(class_images_list)
        class_images_list = [class_path + '/' + image_path for image_path in class_images_list]
        split_number = int(round(0.8 * len(class_images_list)))
        [train_list.append(image_path) for image_path in class_images_list[:split_number]]
        [test_list.append(image_path) for image_path in class_images_list[split_number:]]

    return train_list, test_list


def get_image_generators(config):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                 horizontal_flip=True,
                                                                 rotation_range=30,
                                                                 shear_range=10,
                                                                 width_shift_range=0.15,
                                                                 height_shift_range=0.15,
                                                                 zoom_range=0.2,
                                                                 channel_shift_range=0.2
                                                                 )
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        class_mode='categorical',
        batch_size=config['batch_size'],
        target_size=(IMAGE_DIM, IMAGE_DIM),
        )
    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        class_mode='categorical',
        batch_size=config['batch_size'],
        target_size=(IMAGE_DIM, IMAGE_DIM),
        )
    return train_generator, test_generator


def download_and_prepare_data():
    download_data()
    extract_data()
    train_list, test_list = get_split_data_lists()
    create_data_subset(train_list, 'train')
    create_data_subset(test_list, 'test')
    logger.info(f'Data preparation is done.')



