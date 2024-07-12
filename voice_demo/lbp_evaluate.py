import numpy as np
import os
import random
import shutil
import cv2
import wave
import pyaudio
import logging
import glob
from concurrent.futures import ThreadPoolExecutor
from utils.LBP import LBP,calculate_distance,faster_calculate_LBP
from utils.audio_process import draw_spectrogram
from utils.file_process import remove_files_by_folder,clear_folder


def split_dataset(datasource_folder_path, train_ratio, seed=None, data_augmente=True):
    '''split the dataset into training and testing sets'''
    logging.info("=====================================Split dataset=====================================")
    
    if seed is not None:
        random.seed(seed)

    # create train and test folders
    train_folder = 'train'
    test_folder = 'test'
    if os.path.exists(train_folder):
        clear_folder(train_folder)
    else:
        os.makedirs(train_folder)
    if os.path.exists(test_folder):
        clear_folder(test_folder)
    else:
        os.makedirs(test_folder)
    
    img_dirs = os.listdir(datasource_folder_path)
    for dir in img_dirs:
        root = os.path.join(datasource_folder_path,dir)
        # 数据增强
        if data_augmente:
            data_augmentation(root,ration=0.5)
        train_subfolder = os.path.join(train_folder, dir)
        test_subfolder = os.path.join(test_folder, dir)
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)

        # 按比例划分
        augumented_png_files = [f for f in os.listdir(root)]
        split_index = int(len(augumented_png_files) * train_ratio)
        train_files = augumented_png_files[:split_index]
        test_files = augumented_png_files[split_index:]

        # # 移动文件到对应的目标文件夹
        # for f in train_files:
        #     src_path = os.path.join(datasource_folder_path,dir,f)
        #     dst_path = os.path.join(train_subfolder, f)
        #     shutil.move(src_path, dst_path)
        # for f in test_files:
        #     src_path = os.path.join(datasource_folder_path,dir,f)
        #     dst_path = os.path.join(test_subfolder, f)
        #     shutil.move(src_path, dst_path)

    logging.info("=====================================Finish=====================================")


def process_image(text_name,imgs_path,label,img_type='gray'):
    '''process images and save the LBP feature to a file'''
    count=0
    files= os.listdir(imgs_path)
    with open(text_name, 'a') as f:
        for file in files:
            img_path = os.path.join(imgs_path,file)
            img = cv2.imread(img_path)
            # # vector= LBP(img, label,img_type)
            vector = faster_calculate_LBP(img, label,img_type)
            count+=1
            np.savetxt(f, vector, fmt='%f', delimiter=',')
    logging.info(f"LBP Processed {count} {img_type} images  label:{label}")

def find_key_by_value(dictionary, value) -> str:
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # not found


def evaluate(train_data, test_data, distance_type='L1'):
    '''evaluate the model using test data'''
    accuracy = 0
    for data in test_data:
        distances = []
        for train_vector in train_data:
            distance = calculate_distance(data[1:], train_vector[1:], distance_type)
            distances.append([train_vector[0], distance])
        # sort the distances
        distances = sorted(distances, key=lambda x: x[1])
        np.argmin(distances)

        predicted_label = distances[0][0]
        min_distance = distances[0][1]
        logging.info(f"Predicted label: {predicted_label}, Actual label: {data[0]}, Distance: {min_distance}")
        # print(f"Predicted label: {predicted_label}, Actual label: {data[0]}, Distance: {min_distance}")
        speaker = find_key_by_value(LABEL_VALUES, predicted_label)
        if predicted_label == data[0]:
            accuracy += 1
    accuracy = accuracy / len(test_data)
    logging.info(f"Accuracy: {accuracy}")
    print(f" Accuracy: {accuracy}")
    return predicted_label

def data_augmentation(images_path,ration=0.4):
    '''data augmentation'''
    png_names = [f for f in os.listdir(images_path)]
    if len(png_names)==0:
        return
    # gaussian noise
    random.shuffle(png_names)
    split_index = int(len(png_names) * ration)
    argumenation_images = png_names[:split_index]
    for image in argumenation_images:
        img = cv2.imread(os.path.join(images_path,image))
        # gaussian noise
        noise = np.random.normal(0, 1, img.shape)
        noise_img = img + noise
        noise_img_name=image.replace('.png','_noise.png')
        cv2.imwrite(os.path.join(images_path,noise_img_name),noise_img)

    # Splicing after cutting
    png_names = [f for f in os.listdir(images_path)]
    if len(png_names)==0:
        return
    random.shuffle(png_names)
    image_parts = []
    argumenation_images = png_names[:split_index]
    for image in argumenation_images:
        img = cv2.imread(os.path.join(images_path,image))
        h, w = img.shape[:2]
        part_width = w // 3
        for i in range(3):
            part = img[:, i * part_width:(i + 1) * part_width]
            image_parts.append(part)

    random.shuffle(image_parts)
    for i in range(len(argumenation_images)):
        combined_img = np.hstack((image_parts[i * 3], image_parts[i * 3 + 1], image_parts[i * 3 + 2]))
        # 保存组合后的图像，文件名加上 '_combined' 后缀
        combined_image_name = argumenation_images[i].replace('.png', '_combined.png')
        cv2.imwrite(os.path.join(images_path, combined_image_name), combined_img)
    random.shuffle(image_parts)


if __name__ == '__main__':

    
    TRAIN_RAtIO = 0.8
    IMG_TYPE = ['gray','color']
    VOICE_SOURCE_FOLDER = 'voice_source/'
    VOICE_IMAGES_FOLDER = 'voice_images/'
    LABELS = os.listdir(VOICE_SOURCE_FOLDER)
    LABEL_VALUES = {label: i for i, label in enumerate(LABELS)}

    # 1. convert all wav files in VOICE_SOURCE_FOLDER to png files and save them to the VOICE_IMAGES_FOLDER
    if os.path.exists(VOICE_IMAGES_FOLDER):
        clear_folder(VOICE_IMAGES_FOLDER)
    else:
        os.makedirs(VOICE_IMAGES_FOLDER)
    dirs = os.listdir(VOICE_SOURCE_FOLDER)
    for dir in dirs:
        os.makedirs(os.path.join(VOICE_IMAGES_FOLDER,dir), exist_ok=True)
        files = os.listdir(VOICE_SOURCE_FOLDER + dir)
        for file in files:
            if file.endswith('.wav'):
                draw_spectrogram(VOICE_SOURCE_FOLDER + dir + '/' + file, VOICE_IMAGES_FOLDER + dir + '/' + file.replace('.wav', '.png'))
        logging.info(f"Finish save spectrogram : {dir} to {VOICE_IMAGES_FOLDER + dir}")

    # 2. split the dataset
    split_dataset(VOICE_IMAGES_FOLDER, train_ratio=TRAIN_RAtIO,seed=None,data_augmente=True)

    # # 3. process the images and write the LBP features to a file
    # for img_type in IMG_TYPE:
    #     for split in ['train', 'test']:
    #         TXT_FILE = f'{split}_{img_type}.txt'
    #         with open(TXT_FILE, 'w') as file:
    #             pass  # clear the file
    #         for label in LABELS:
    #             # image_paths = glob.glob(os.path.join(f'./{split}/{label}/', '*'))
    #             image_paths = os.path.join(split, label)
    #             process_image(TXT_FILE,image_paths, LABEL_VALUES[label], img_type=img_type)
    #         print(f"Finish process {split} {label} {img_type} images")
    
    


    # 3. process the images and write the LBP features to a file
    for img_type in IMG_TYPE:
        TXT_FILE = f'model/{img_type}_all.txt'
        with open(TXT_FILE, 'w') as file:
            pass  # clear the file
        for label in LABELS:
            # image_paths = glob.glob(os.path.join(f'./{split}/{label}/', '*'))
            image_paths = os.path.join(VOICE_IMAGES_FOLDER, label)
            process_image(TXT_FILE,image_paths, LABEL_VALUES[label], img_type=img_type)
            print(f"Finish process {label} {img_type} images")

    # for dir in dirs:
    #     # os.makedirs(os.path.join(VOICE_IMAGES_FOLDER,dir), exist_ok=True)
    #     files = os.listdir(VOICE_SOURCE_FOLDER + dir)
    #     count=0
    #     for file in files:
    #         if file.endswith('.wav'):
    #             #rename the wav file 数字
    #             new_file = str(count)+'_'+dir
    #             try:
    #                 os.rename(VOICE_SOURCE_FOLDER + dir + '/' + file,VOICE_SOURCE_FOLDER + dir + '/' + new_file + '.wav')
    #             except:
    #                 pass
    #             count+=1
        

    
    # # 4. evaluate the model
    # for img_type in IMG_TYPE:
    #     train_data = np.loadtxt(f'model/train_{img_type}.txt', delimiter=',')
    #     test_data = np.loadtxt(f'model/test_{img_type}.txt', delimiter=',')
    #     logging.info("=====================================image_type:"+img_type+"=====================================")
    #     print(train_data.shape)
    #     print(test_data.shape)
    #     evaluate(train_data, test_data, distance_type='L1')