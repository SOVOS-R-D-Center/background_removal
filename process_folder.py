import os
import cv2
from tqdm import tqdm
from image_folder import make_dataset
from selfieSegmentation import MPSegmentation

if __name__ == '__main__':
    segmentationModule = MPSegmentation(threshold=0.4, bg_color=(0,255,0))

    inp_folder = '/home/daniel/Documents/temp/selfie_samples/'
    opt_folder = '/home/daniel/Documents/temp/selfie_white_bgn/'

    dataset =  make_dataset(inp_folder)

    for path in tqdm(dataset):
        name = path.split('/')[-1]
        opt_path = os.path.join(opt_folder, name)

        img1 = cv2.imread(path)

        img2 = segmentationModule(img1)

        cv2.imwrite(opt_path, img2)
