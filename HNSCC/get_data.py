import os
import math

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class SurvivalDataset(Dataset):
    def __init__(self):
        # Find a better way to flip some of these images to the correct orientation
        self.flip_array = [
            'data/HNSCC-01-0199/10-26-2002-RT SIMULATION-18560/3.000000-58373',
            'data/HNSCC-01-0018/03-01-2009-RT SIMULATION-16942/10.000000-60812',
            'data/HNSCC-01-0011/11-11-1997-RT SIMULATION-19590/10.000000-88887',
            'data/HNSCC-01-0016/03-07-1998-RT SIMULATION-64055/10.000000-15759',
            'data/HNSCC-01-0086/07-03-1999-RT SIMULATION-42088/92691',
            'data/HNSCC-01-0017/05-04-1998-RT SIMULATION-38793/10.000000-76074',
            'data/HNSCC-01-0010/03-30-2009-RT SIMULATION-52386/10.000000-61881',
            'data/HNSCC-01-0214/07-27-2004-RT SIMULATION-67763/3.000000-43635',
            'data/HNSCC-01-0004/08-24-1996-RT SIMULATION-72882/10.000000-74926',
            'data/HNSCC-01-0201/10-21-2002-RT SIMULATION-79781/2.000000-47027',
            'data/HNSCC-01-0005/01-19-1998-RT SIMULATION-39358/10.000000-79230',
            'data/HNSCC-01-0208/02-02-2003-RT SIMULATION-27554/6.000000-44043',
            'data/HNSCC-01-0183/07-01-2002-RT SIMULATION-37569/3.000000-83360',
            'data/HNSCC-01-0015/02-16-1998-RT SIMULATION-83066/10.000000-53517',
            'data/HNSCC-01-0012/10-11-1997-RT SIMULATION-86133/10.000000-11509',
            'data/HNSCC-01-0013/02-23-1998-RT SIMULATION-65901/10.000000-22461',
            'data/HNSCC-01-0085/09-21-1999-RT SIMULATION-14635/2.000000-33945',
            'data/HNSCC-01-0007/04-29-1997-RT SIMULATION-32176/10.000000-72029',
            'data/HNSCC-01-0009/05-14-1997-RT SIMULATION-35793/10.000000-66060',
            'data/HNSCC-01-0203/12-10-2002-RT SIMULATION-76477/4.000000-83318',
            'data/HNSCC-01-0008/05-14-1997-RT SIMULATION-54670/10.000000-96097',
            'data/HNSCC-01-0006/06-09-1997-RT SIMULATION-57410/10.000000-41643',
            'data/HNSCC-01-0180/05-15-2002-RT SIMULATION-79756/3.000000-98866',
            'data/HNSCC-01-0188/08-13-2002-RT SIMULATION-59772/5.000000-29932',
            'data/HNSCC-01-0175/06-04-2002-RT SIMULATION-40000/3.000000-91467'
        ]
        # Load the data file
        TREATMENT_FILE = 'Patient-and-Treatment-Characteristics.xls'
        treatment_data = pd.read_excel(TREATMENT_FILE)
        # Get the valid dates and patients
        valid_paths, valid_patients, valid_pixels = self.get_dates()
        # Assign class attributes
        treatment_data = treatment_data[treatment_data['TCIA code'].isin(valid_patients)]
        # Convert TCIA codes to indices
        self.all_patients = valid_patients #treatment_data['TCIA code'].tolist()
        self.num_patients = len(valid_patients) #treatment_data.shape[0]
        self.patient = torch.tensor(list(range(self.num_patients)))
        # Set all the image file paths and image arrays
        self.all_paths = valid_paths
        self.all_axial = [self.crop_image(p,valid_pixels[i],'axial') for i,p in enumerate(self.all_paths)]
        self.all_coronal = [self.crop_image(p,valid_pixels[i],'coronal') for i,p in enumerate(self.all_paths)]
        self.all_sagittal = [self.crop_image(p,valid_pixels[i],'sagittal') for i,p in enumerate(self.all_paths)]
        # Set Dead = 0 and Alive = 1
        self.outcome = torch.tensor([0 if o=='Dead' else 1 for o in treatment_data['Alive or Dead'].tolist()])

    def __getitem__(self, index):
        """
        Return the TCIA code of the patient, the axial scan, coronal scan,
        saggital scan, and their outcome (Dead = 0, Alive = 1).
        """
        return (self.patient[index], self.all_axial[index],
                self.all_coronal[index], self.all_sagittal[index],
                self.outcome[index])

    def __len__(self):
        return self.num_patients

    def get_dates(self):
        valid_paths = []
        valid_patients = []
        valid_pixels = []
        all_patients = os.listdir('data')
        all_patients = [p for p in all_patients if '.DS_Store' not in p]
        for p in all_patients:
            patient_dir = os.path.join('data', p)
            scan_dates = [d for d in os.listdir(patient_dir) if 'RT' in d]
            scan_dates = [d for d in scan_dates if '.DS_Store' not in d]
            # Some patients have multiple scans, the first should be the best
            if len(scan_dates) > 1:
                scan_dates = scan_dates[:1]
            for s in scan_dates:
                all_files = os.listdir(os.path.join('data', p, s))
                all_files.sort()
                ct_file = all_files[-1]
                path = os.path.join('data', p, s, ct_file)
                try:
                    patient_dicom = self.load_scan(path)
                    patient_pixels = self.get_pixels_hu(patient_dicom)
                    valid_paths.append(path)
                    valid_patients.append(p)
                    valid_pixels.append(patient_pixels)
                except:
                    pass
        return valid_paths, valid_patients, valid_pixels

    def crop_image(self, path, patient_pixels, plane):
        """
        Ideally remove this once the images are generated and put into the
        data folder.
        """
        if path in self.flip_array:
            patient_pixels = patient_pixels[::-1]
        # Get the middle/uppermost 128x128 patch
        patch_size = 128
        middle_slice_x = patient_pixels.shape[1]//2
        start_ind_x = middle_slice_x - patch_size//2
        stop_ind_x = middle_slice_x + patch_size//2
        middle_slice_y = patient_pixels.shape[2]//2
        start_ind_y = middle_slice_y - patch_size//2
        stop_ind_y = middle_slice_y + patch_size//2
        if patient_pixels.shape[0] > patch_size:
            middle_slice_z = patient_pixels.shape[0] - patch_size//2
            start_slice_z = patient_pixels.shape[0] - patch_size
        else:
            middle_slice_z = patient_pixels.shape[0]//2
            start_slice_z = 0
        # Get the correct image data based on the plane
        if plane == 'axial':
            img_data = patient_pixels[middle_slice_z,
                                      start_ind_x:stop_ind_x,
                                      start_ind_y:stop_ind_y]
            img_data = np.pad(img_data,
                              (((patch_size-img_data.shape[0])//2,
                                (patch_size-img_data.shape[0])//2),
                               ((patch_size-img_data.shape[1])//2,
                                (patch_size-img_data.shape[1])//2)),
                              'constant',
                              constant_values=np.min(img_data))
        if plane == 'coronal':
            img_data = patient_pixels[start_slice_z:,
                                      middle_slice_x,
                                      start_ind_y:stop_ind_y]
            img_data = np.pad(img_data,
                              (((patch_size-img_data.shape[0])//2,
                                (patch_size-img_data.shape[0])//2),
                               ((patch_size-img_data.shape[1])//2,
                                (patch_size-img_data.shape[1])//2)),
                              'constant',
                              constant_values=np.min(img_data))
        if plane == 'sagittal':
            img_data = patient_pixels[start_slice_z:,
                                      start_ind_x:stop_ind_x,
                                      middle_slice_y]
            img_data = np.pad(img_data,
                              (((patch_size-img_data.shape[0])//2,
                                (patch_size-img_data.shape[0])//2),
                               ((patch_size-img_data.shape[1])//2,
                                (patch_size-img_data.shape[1])//2)),
                              'constant',
                              constant_values=np.min(img_data))
        return torch.from_numpy(img_data)

    def load_scan(self, path):
        slices = [pydicom.dcmread(path + os.sep + s) for s in               
                  os.listdir(path)]
        slices = [s for s in slices if 'SliceLocation' in s]
        slices.sort(key = lambda x: int(x.InstanceNumber))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] -
                                     slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation -
                                     slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    def get_pixels_hu(self, scans):
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)    
        return np.array(image, dtype=np.int16)


if __name__ == '__main__':
    batch_size = 5
    dataset = SurvivalDataset()
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    data_iter = iter(data_loader)
    (p, a, c, s, o) =  data_iter.next()

    def imshow(tensor_img):
        np_img = tensor_img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(a, nrow=batch_size))
    imshow(torchvision.utils.make_grid(c, nrow=batch_size))
    imshow(torchvision.utils.make_grid(s, nrow=batch_size))
    print(' '.join([str(o[j]) for j in range(batch_size)]))


    # # Determine if they are alive or dead
    # alive_dead = [treatment_data[treatment_data['TCIA code'] == p]['Alive or Dead'].values[0] for p in valid_patients]
    # unique, counts = np.unique(alive_dead, return_counts=True)
    # alive_dead_counts = dict(np.asarray((unique, counts)).T)
    # alive_dead_counts = dict([a, int(x)] for a, x in alive_dead_counts.items())
    # # Divide into train and test sets
    # train_percent = 0.75
    # train_set = {
    #     'Alive': round(alive_dead_counts['Alive'] * train_percent),
    #     'Dead': round(alive_dead_counts['Dead'] * train_percent),
    # }
    # test_set = {
    #     'Alive': round(alive_dead_counts['Alive'] * (1-train_percent)),
    #     'Dead': round(alive_dead_counts['Dead'] * (1-train_percent)),
    # }
    # # random.shuffle(valid_patients)
    # partition = {
    #     'train': ['id-1', 'id-2', 'id-3']
    # }
    # # Assign 0 to be dead and 1 to be alive using `Alive or Dead` variable
    # labels = dict([p,alive_dead[i]] for i,p in enumerate(valid_patients))