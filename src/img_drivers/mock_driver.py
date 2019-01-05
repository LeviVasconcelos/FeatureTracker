import cv2
from fm import io 

class FromFiles:
    def __init__(self, folder_path, img_ext):
        self.file_paths, n = io.GetFilePaths(folder_path, img_ext)
        self.current_index = 0
        if (n == 0):
            print('no images loaded from: %s/*.%s' % (folder_path, img_ext))
        self.last_img_idx = n
    def next(self, number = 1):
        images = []
        end_files = False
        if (self.current_index + number) > (self.last_img_idx - 1):
            number = self.last_img_idx - self.current_index
            end_files = True
        for i in range(self.current_index, self.current_index + number):
            img = cv2.imread(self.file_paths[i], cv2.IMREAD_GRAYSCALE)
            images.append(img)
        self.current_index += (number - 1)
        return images, end_files


