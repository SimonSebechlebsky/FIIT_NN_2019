import os
import requests
import shutil
import logging
import scipy.io as sio
import cv2
from xml.dom import minidom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'dog_breeds_data')


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
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar' , DATA_DIR, 'images.tar')
    if not os.path.exists(os.path.join(DATA_DIR, 'annotation.tar')):
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar' ,DATA_DIR, 'annotation.tar')
    if not os.path.exists(os.path.join(DATA_DIR, 'lists.tar')):
        download_file('http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar' ,DATA_DIR, 'lists.tar')


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


def crop_image(annotation_path, src):
    dom = minidom.parse(annotation_path)
    object_tag = dom.getElementsByTagName('object')
    bndbox_tag = object_tag[0].getElementsByTagName('bndbox')
    xmin = int(bndbox_tag[0].getElementsByTagName('xmin')[0].firstChild.nodeValue)
    ymin = int(bndbox_tag[0].getElementsByTagName('ymin')[0].firstChild.nodeValue)
    xmax = int(bndbox_tag[0].getElementsByTagName('xmax')[0].firstChild.nodeValue)
    ymax = int(bndbox_tag[0].getElementsByTagName('ymax')[0].firstChild.nodeValue)
    img = cv2.imread(src)
    cropped_img = img[ymin:ymax, xmin:xmax]
    return cropped_img


def create_data_subset(image_list, destination_folder):
    logger.info(f'Creating {destination_folder} subset...')
    images_folder = prefix_with_data_folder('Images')
    annotations_folder = prefix_with_data_folder('Annotation')
    ensure_existing_data_folder(destination_folder)
    for image_path in image_list['file_list']:
        path_str = image_path[0][0]
        src = os.path.join(images_folder, path_str)
        dst = os.path.join(prefix_with_data_folder(destination_folder), path_str)
        class_folder = path_str.split('/')[0]
        ensure_existing_data_folder(os.path.join(destination_folder, class_folder))
        annotation_str = path_str.split('.')[0]
        annotation_path = os.path.join(annotations_folder, annotation_str)
        cropped_img = crop_image(annotation_path, src)
        cv2.imwrite(dst, cropped_img)


def get_split_data_lists():
    train_list = sio.loadmat(prefix_with_data_folder('Lists/train_list.mat'))
    test_list = sio.loadmat(prefix_with_data_folder('Lists/test_list.mat'))
    return train_list, test_list


def download_and_prepare_data():
    download_data()
    extract_data()
    train_list, test_list = get_split_data_lists()
    create_data_subset(train_list, 'train')
    create_data_subset(test_list, 'test')
    logger.info(f'Data preparation is done.')



