import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import shutil
from numba import jit


def split_dataset(folder_path, train_ratio=0.8, seed=None):
    '''
    split the dataset into training and testing sets
    
    '''
    print("=====================================Split dataset=====================================")
    if seed is not None:
        random.seed(seed)

    # define source folder
    source_folder = os.path.join(folder_path, 'A')
    
    # define target folders
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')
    
    # create target folders
    for split in ['train', 'test']:
        for category in ['busy', 'free']:
            os.makedirs(os.path.join(folder_path, split, category), exist_ok=True)

    # list all files in source folder
    for category in ['busy', 'free']:
        category_path = os.path.join(source_folder, category)
        all_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        print(all_files)
        # calculate train size
        train_size = int(len(all_files) * train_ratio)

        # shuffle files
        random.shuffle(all_files)
        train_files = all_files[:train_size]
        test_files = all_files[train_size:]

        # move files to target folders
        for file in train_files:
            shutil.move(os.path.join(category_path, file), os.path.join(train_folder, category, file))

        for file in test_files:
            shutil.move(os.path.join(category_path, file), os.path.join(test_folder, category, file))

        print(f"Total files in {category}: {len(all_files)}")
        print(f"Training files in {category}: {len(train_files)}")
        print(f"Testing files in {category}: {len(test_files)}")
    print("=====================================Split dataset completed=====================================")

def LBP(image,label,img_type):
    '''
    implement LBP
    '''
    # 1. convert image
    if img_type == 'gray':
        read_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif img_type == 'color':
        blue_channel, green_channel, red_channel = cv2.split(image)
        read_img = np.concatenate((blue_channel, green_channel, red_channel), axis=1)
    else:
        read_img=image

    vector = np.zeros(read_img.shape, dtype=np.uint8)
    # 2. calculate LBP
    for i in range(1,read_img.shape[0]-1 ):
        for j in range(1, read_img.shape[1]-1):
            # 3*3 core
            # binary_pattern = ((neighbor >= center) * 1).astype(np.uint8)
            # make a binary pattern
            binary_pattern = np.array([
                1 if read_img[i-1, j-1]>read_img[i, j] else 0,  # 左上角 
                1 if read_img[i-1, j]> read_img[i, j] else 0,  # 上边
                1 if read_img[i-1, j+1]> read_img[i, j] else 0,  # 上边
                1 if read_img[i, j+1]> read_img[i, j] else 0, 
                1 if read_img[i+1, j+1]> read_img[i, j] else 0,  # 右边
                1 if read_img[i+1, j]> read_img[i, j] else 0, 
                1 if read_img[i+1, j-1]> read_img[i, j] else 0,  # 下边
                1 if read_img[i, j-1]> read_img[i, j] else 0  # 左边
            ])
            # convert binary pattern to decimal
            pattern = binary_pattern.dot(2 ** np.arange(binary_pattern.size)[::-1])
            # print(pattern)    
            vector[i,j]=pattern
    # 3. count the frequency of each pixel value
    hist = np.bincount(vector.flatten(), minlength=256).astype(np.float32)
    # hist = hist / np.sum(hist)
    return np.insert(np.array([hist.flatten()]).astype(np.float32), 0, label).reshape(1, -1)


def process_image(file_name,imgs_path,label,img_num,img_type='gray'):
    '''process images and save the LBP feature to a file'''
    count=0
    with open(file_name, 'a') as file:
        for image_path in imgs_path:
            img = cv2.imread(image_path)
            # if img.shape!=(150,150,3):
            #     continue
            count+=1
            vector= LBP(img, label,img_type)
            np.savetxt(file, vector, fmt='%f', delimiter=',')
            if count % img_num == 0:
                break
    print(f"LBP Processed {count} {img_type} images  label:{label}")


def caculate_distance(vector1, vector2, distance_type='L1'):
    '''caculate the distance between two vectors'''
    if distance_type == 'L1':
        return np.sum(np.abs(vector1 - vector2))
    elif distance_type == 'L2':
        return np.sqrt(np.sum(np.square(vector1 - vector2)))
    elif distance_type == 'cosine':
        return 1-np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif distance_type == 'Euclidean':
        return np.linalg.norm(vector1 - vector2)
    else:
        raise Exception("Invalid distance type")


def evaluate(train_data, test_data):
    '''evaluate the model using test data'''
    accuracy = 0
    for data in test_data:
        distances = []
        for train_vector in train_data:
            distance = caculate_distance(data[1:], train_vector[1:], 'cosine')
            distances.append([train_vector[0], distance])
        # sort the distances
        distances = sorted(distances, key=lambda x: x[1])
        np.argmin(distances)

        predicted_label = distances[0][0]
        min_distance = distances[0][1]
        
        print(f"Predicted label: {predicted_label}, True label: {data[0]}, Distance: {min_distance},{'Correct' if predicted_label == data[0] else 'Wrong'}")
        if predicted_label == data[0]:
            accuracy += 1
    accuracy = accuracy / len(test_data)
    print(f"=====================================Accuracy: {accuracy}=====================================")



if __name__ == '__main__':

    LABEL={
        'busy':1,
        'free':0
    }
    IMG_NUM = 500
    TRAIN_RAtIO = 0.8
    IMG_TYPE = ['gray','color']

    # split the dataset from folder A
    current_folder_path = os.getcwd()  # get the current folder path
    # split_dataset(current_folder_path, train_ratio=TRAIN_RAtIO, seed=42)

    for img_type in IMG_TYPE:
        for split in ['train', 'test']:
            for category in ['busy', 'free']:
                TXT_FILE = f'{split}_{img_type}.txt'
                # if os.path.exists(TXT_FILE):
                #     with open(TXT_FILE, 'w') as file:
                #         file.write('')
                image_paths = glob.glob(os.path.join(f'./{split}/{category}/', '*'))
                process_image(TXT_FILE,image_paths, LABEL[category], img_num=IMG_NUM, img_type=img_type)

    # # train data write to file
    # train_image_paths_busy = glob.glob(os.path.join('./train/busy/', '*'))
    # train_image_paths_free = glob.glob(os.path.join('./train/free/', '*'))
    # process_image("train_gray.txt",train_image_paths_busy, BUSY_LABEL, img_num=50,img_type=IMG_TYPE)
    # process_image("train_gray.txt",train_image_paths_free, FREE_LABEL, img_num=50,img_type=IMG_TYPE)
    # # test data write to file
    # test_image_paths_busy = glob.glob(os.path.join('./test/busy/', '*'))
    # test_image_paths_free = glob.glob(os.path.join('./test/free/', '*'))
    # process_image("test_gray.txt",test_image_paths_busy, BUSY_LABEL, img_num=50,img_type=IMG_TYPE)
    # process_image("test_gray.txt",test_image_paths_free, FREE_LABEL, img_num=50,img_type=IMG_TYPE)
    
    for img_type in IMG_TYPE:
        train_data = np.loadtxt(f'train_{img_type}.txt', delimiter=',')
        test_data = np.loadtxt(f'test_{img_type}.txt', delimiter=',')
        print("=====================================image_type:"+img_type+"=====================================")
        evaluate(train_data, test_data)
    