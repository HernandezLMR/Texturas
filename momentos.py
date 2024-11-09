import cv2
import os
import numpy as np
from scipy.stats import skew, kurtosis

class Instance:
    def __init__(self, path):
        self.path = path
        self.median = None
        self.variance = None
        self.asymmetry = None
        self.kurtosis = None
        self.image = None
        self.imageBW = None

    def get_image(self, path):
        # Load the image using PIL
        try:
            self.image = cv2.imread(path, cv2.IMREAD_COLOR)
            self.imageBW = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return True
        except IOError:
            print(f"Error: {path} is not a valid image file.")
            return False

    def calculate_moments(self):
        # Calculate moments
        if self.image is not None:
            self.median = np.median(self.imageBW)
            self.variance = np.var(self.imageBW)
            image_flat = self.imageBW.flatten()
            self.asymmetry = skew(image_flat)
            self.kurtosis = kurtosis(image_flat)

    @classmethod
    def load_images_from_directory(cls, directory_path):
        instances = []
        for filename in os.listdir(directory_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):  # You can expand the file types
                image_path = os.path.join(directory_path, filename)
                instance = cls(image_path)
                if instance.get_image():
                    instance.calculate_moments()
                    instances.append(instance)
        return instances

if __name__ == '__main__':
    # Example usage
    instances = []
    directory_path = "./DS1"
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg"):
            instance = Instance(os.path.join(directory_path, filename))
            if instance.get_image(instance.path):
                instance.calculate_moments()
                instances.append(instance)
    
    for i in instances:
        if i.image is not None:
            cv2.imshow('image', i.image)
            print(i.median)
            print(i.variance)
            print(i.asymmetry)
            print(i.kurtosis)
            cv2.waitKey(0)


