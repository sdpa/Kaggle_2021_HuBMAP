from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import tifffile as tiff
# import matplotlib.pyplot as plt


class HuBMAPDataset(Dataset):
    def __init__(self, base_dir, mode):
        self.base_dir = base_dir
        self.mode = mode
        self.images = []
        self.patient_masks = {}  # The masks for 9 training patients.
        self.img_dim = {}  # Height and width of tiff file for each patient.
        # Get all patients masks.
        patients = pd.read_csv(base_dir + "/train.csv")
        dataset_info = pd.read_csv(base_dir + "/HuBMAP-20-dataset_information.csv")
        # print(dataset_info)
        for i in range(len(patients)):
            patientName = patients['id'].iloc[i]
            patient_metadata = dataset_info.loc[dataset_info["image_file"] == patientName + ".tiff"]
            height = patient_metadata["height_pixels"].iloc[0]
            width = patient_metadata["width_pixels"].iloc[0]
            self.patient_masks[patientName] = self.rle_to_mask(patients['encoding'].iloc[i], (height, width))
            grid = self.make_grid((height, width), window=1024)
            _id = [patientName]*grid.shape[0]
            _id = np.array(_id).reshape(grid.shape[0], 1)
            # Concatenate patient name
            grid = np.concatenate((grid, np.array(_id)), axis = 1)
            if len(self.images) > 0:
                self.images = np.concatenate((self.images, grid), axis = 0)
            else:
                self.images = grid

            # print(self.patient_masks)
            # plt.imshow(self.patient_masks[patientName])
            # plt.show()

    def rle_to_mask(self, encoding, size):
        # encoding is a string. convert it to int.
        mask = np.zeros(size[0] * size[1], dtype=np.uint8)
        encoding = np.fromstring(encoding, dtype=int, sep=' ')
        starts = np.array(encoding)[::2]
        lengths = np.array(encoding)[1::2]
        for start, length in zip(starts, lengths):
            mask[start:start + length] = 1
        return mask.reshape(size, order='F')

    def make_grid(self, shape, window=1024, min_overlap=32):
        """
            Return Array of size (N,4), where N - number of tiles,
            2nd axis represente slices: x1,x2,y1,y2
        """
        y, x = shape
        nx = x // (window - min_overlap) + 1
        x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
        x1[-1] = x - window
        x2 = (x1 + window).clip(0, x)
        ny = y // (window - min_overlap) + 1
        y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
        y1[-1] = y - window
        y2 = (y1 + window).clip(0, y)
        slices = np.zeros((nx, ny, 4), dtype=np.int64)

        for i in range(nx):
            for j in range(ny):
                slices[i, j] = x1[i], x2[i], y1[j], y2[j]
        return slices.reshape(nx * ny, 4)

    def __getitem__(self, index):
        x1, x2, y1, y2, patient = self.images[index]
        patient_img = tiff.imread(self.base_dir + "/" + patient + '.tiff')
        print("Before Tiff file dim: ", patient_img.shape)
        if(len(patient_img.shape) > 3):
            patient_img = patient_img.squeeze(0).squeeze(0)
            patient_img = np.moveaxis(patient_img, 0, -1)
        print("After Tiff file dim: ", patient_img.shape)
        print("--"*20)
        slice_img = patient_img[int(y1):int(y2),int(x1):int(x2)]
        slice_mask = self.patient_masks[str(self.images[index][-1])][int(y1):int(y2),int(x1):int(x2)]
        return slice_img, slice_mask

    def __len__(self):
        return len(self.images)
