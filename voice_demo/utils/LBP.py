import cv2
import numpy as np
from scipy.spatial import distance
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

def faster_calculate_LBP(image: np.ndarray,label,img_type) -> np.ndarray:
    '''calculate the LBP feature of the image'''
    # 1. convert image
    if img_type == 'gray':
        read_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif img_type == 'color':
        blue_channel, green_channel, red_channel = cv2.split(image)
        read_img = np.concatenate((blue_channel, green_channel, red_channel), axis=1)
    else:
        read_img=image

    padded_image = np.pad(read_img, pad_width=1, mode='edge')
    lbp_image = np.zeros_like(read_img, dtype=np.int32)

    # Define the offsets for the 8 neighbors
    offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]

    for idx, (di, dj) in enumerate(offsets):
        # Shift the padded image using the offsets
        shifted_image = padded_image[1 + di: 1 + di + image.shape[0], 1 + dj: 1 + dj + read_img.shape[1]]
        # Update the binary representation of the LBP image
        lbp_image += (shifted_image >= padded_image[1:-1, 1:-1]) << idx

    hist = np.bincount(lbp_image.flatten(), minlength=256).astype(np.float32)
    # hist = hist / np.sum(hist)
    # return lbp_image.astype(np.uint8)
    return np.insert(np.array([hist.flatten()]).astype(np.float32), 0, label).reshape(1, -1)


def calculate_distance(vector1, vector2, distance_type='L1'):
    '''Calculate the distance between two vectors'''
    if distance_type == 'L1':
        return np.sum(np.abs(vector1 - vector2))
    elif distance_type == 'L2':
        return np.sqrt(np.sum(np.square(vector1 - vector2)))
    elif distance_type == 'cosine':
        return 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    elif distance_type == 'Euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif distance_type == 'Chebyshev':
        return np.max(np.abs(vector1 - vector2))
    elif distance_type == 'Minkowski':
        p = 3  # You can change the order of the Minkowski distance
        return np.power(np.sum(np.abs(vector1 - vector2) ** p), 1 / p)
    elif distance_type == 'Hamming':
        return np.sum(vector1 != vector2) / len(vector1)
    elif distance_type == 'Jaccard':
        return distance.jaccard(vector1, vector2)
    elif distance_type == 'BrayCurtis':
        return distance.braycurtis(vector1, vector2)
    else:
        raise Exception("Invalid distance type")