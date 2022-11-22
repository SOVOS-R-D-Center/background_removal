import os
import cv2
from tqdm import tqdm
from image_folder import make_dataset
from selfieSegmentation import MPSegmentation
from argparse import ArgumentParser as argparse

if __name__ == '__main__':
    # Parse arguments
    parser = argparse()
    parser.add_argument('-i', '--inp_folder', default='/home/daniel/Documents/temp/selfie_samples/',
                                            help='Folder with face images.')
    parser.add_argument('-o', '--opt_folder', default='/home/daniel/Documents/temp/selfie_white_bgn/',
                                            help='Where to save processed images.')
    parser.add_argument('-t', '--threshold', type=int, default=0.3,
                                            help='Mask processing treshold.')
    args = parser.parse_args()

    # Make Face sgmentation object
    segmentationModule = MPSegmentation(threshold=args.threshold, bg_color=(255,255,255))

    # Read image names
    dataset =  make_dataset(args.inp_folder)

    for path in tqdm(dataset):
        # Get I/O names
        name = path.split('/')[-1]
        opt_path = os.path.join(args.opt_folder, name)

        # Read input image
        img1 = cv2.imread(path)

        # Change background
        img2 = segmentationModule(img1)

        # Save output image
        cv2.imwrite(opt_path, img2)
