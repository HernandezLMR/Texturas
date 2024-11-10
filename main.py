import cv2
import numpy as np
from momentos import Instance

def main():
    #Load all images
    directory_path = "./DS1"
    instances = Instance.load_images_from_directory(directory_path)

    for i in instances:
        if i.image is not None:
            cv2.imshow('image', i.image)
            print("///////////////////////")
            print(f"Media: {i.median}")
            print(f"Varianza: {i.variance}")
            print(f"Asimetria: {i.asymmetry}")
            print(f"Curtosis: {i.kurtosis}")
            print(f"Contraste: {i.contrast}")
            print(f"Homogeneidad: {i.homogeneity}")
            print(f"Energia: {i.energy}")
            print(f"Entropia: {i.entropy}")
            print("///////////////////////")
            cv2.waitKey(0)

if __name__ == '__main__':
    main()