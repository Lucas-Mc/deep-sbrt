import os
import math
import random

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from skimage import io, transform
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class SurvivalDataset(Dataset):
    def __init__(self, subset, train_split, transform=None):
        """
        Parameters
        ----------
        subset : str
            'train' or 'test'; splits data into even cohorts.
        train_split : float
            The proportion of total data allocated for training.
            Range = [0,1].
        transfrom : torchvision.transforms, optional
            How to manipulate the data, if desired.
        """
        # Load the data file
        self.treatment_data = pd.read_excel('Patient-and-Treatment-Characteristics.xls')
        # Convert TCIA codes to indices
        self.all_patients = self.get_patients(subset, train_split)
        self.num_patients = len(self.all_patients)
        self.patients = np.array(range(self.num_patients))
        # Set all the image file paths and image arrays
        self.all_axial = np.array([self.get_image(p,'axial') for p in self.all_patients])
        self.all_coronal = np.array([self.get_image(p,'coronal') for p in self.all_patients])
        self.all_sagittal = np.array([self.get_image(p,'sagittal') for p in self.all_patients])
        # Set Dead = 1 and Alive = 0
        self.outcome = np.array([1 if self.patient_outcome(p)=='Dead' else 0 for p in self.all_patients])
        # Save the transform
        self.transform = transform

    def __getitem__(self, index):
        """
        Return the TCIA code of the patient, the axial scan, coronal scan,
        sagittal scan, and their outcome (Dead = 1, Alive = 0).

        Parameters
        ----------
        index : int
            The desired patient index.
        """
        out_data = (self.patients[index], self.all_axial[index],
                    self.all_coronal[index], self.all_sagittal[index],
                    self.outcome[index])
        if self.transform:
            return self.transform(out_data)
        return out_data


    def __len__(self):
        """
        """
        return self.num_patients

    def get_patients(self, subset, train_split):
        """
        Parameters
        ----------
        subset : str
            'train' or 'test'; splits data into even cohorts.
        train_split : float
            The proportion of total data allocated for training.
            Range = [0,1].
        """
        # Get all the patients and split by their outcome
        valid_patients = [f for f in os.listdir('images') if f.startswith('HNSCC')]
        valid_patients.sort()
        all_outcomes = [self.patient_outcome(p) for p in valid_patients]
        all_alive = [(i,o) for i,o in enumerate(all_outcomes) if o=='Alive']
        all_dead = [(i,o) for i,o in enumerate(all_outcomes) if o=='Dead']
        # Randomly shuffle both lists using the same seed for reproducibility
        random.Random(4).shuffle(all_alive)
        random.Random(4).shuffle(all_dead)
        alive_split = int(np.floor(train_split * len(all_alive)))
        dead_split = int(np.floor(train_split * len(all_dead)))
        if subset == 'train':
            all_alive = all_alive[:alive_split]
            all_dead = all_dead[:dead_split]
        elif subset == 'test':
            all_alive = all_alive[alive_split:]
            all_dead = all_dead[dead_split:]
        else:
            raise Exception('Invalid subset provided: Must be "train" or "test"')
        # Combine the alive and dead data subsets, then randomly shuffle
        combined_idx = [a[0] for a in all_alive]
        combined_idx.extend([d[0] for d in all_dead])
        combined_patients = [valid_patients[i] for i in combined_idx]
        random.Random(1).shuffle(combined_patients)
        return combined_patients

    def patient_outcome(self, patient):
        """
        Get the outcome of a patient.

        Returns
        -------
        patient_outcome : str
            Either 'Alive' or 'Dead' based on the patient.

        """
        # Assign class attributes
        patient_outcome = self.treatment_data[self.treatment_data['TCIA code']==patient]['Alive or Dead'].values[0]
        return patient_outcome

    def get_image(self, patient, plane):
        """
        Parameters
        ----------
        patient : str
        plane : str
            Either 'axial', 'coronal', or 'sagittal'.

        """
        image_path = os.path.join('images', patient, plane+'.jpg')
        image = io.imread(image_path, as_gray=True)
        return image

    def plot_image(self, images, batch_size=1):
        """
        Plot either a single image or batch of images.

        Parameters
        ----------
        images : list
        batch_size : int

        """
        images = images.numpy().transpose((0, 2, 3, 1))
        fig = plt.figure(figsize=(8,8))
        for i in range(batch_size):         
            ax = plt.subplot(1,batch_size,i+1)
            ax.axis('off')
            plt.imshow(images[i], cmap=plt.get_cmap('bone'))
        plt.tight_layout()
        plt.show()


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Parameters
    ----------
    output_size : tuple or int
        Desired output size. If tuple, output is matched to output_size. If
        int, smaller of image edges is matched to output_size keeping aspect
        ratio the same.

    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        (p, a, c, s, o) = sample
        a = self.transform_image(a)
        c = self.transform_image(c)
        s = self.transform_image(s)
        return (p, a, c, s, o)

    def transform_image(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        return image


class ToTensor(object):
    """
    Convert numpy arrays to Tensors.

    """
    def __call__(self, sample):
        (p, a, c, s, o) = sample
        # Add a dimension for gray-scale images
        a = np.expand_dims(a, axis=2)
        c = np.expand_dims(c, axis=2)
        s = np.expand_dims(s, axis=2)
        # Swap color axis because:
        # numpy image: H x W x C
        # torch image: C X H X W
        a = a.transpose((2, 0, 1))
        c = c.transpose((2, 0, 1))
        s = s.transpose((2, 0, 1))
        return (p, torch.from_numpy(a), torch.from_numpy(c),
                torch.from_numpy(s), o)


if __name__ == '__main__':
    batch_size = 5
    dataset = SurvivalDataset(
        transform=torchvision.transforms.Compose([Rescale(128), ToTensor()])
    )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    data_iter = iter(data_loader)

    fig = plt.figure(figsize=(8,8))
    (p, a, c, s, o) =  data_iter.next()
    a = a.numpy().transpose((0, 2, 3, 1))
    c = c.numpy().transpose((0, 2, 3, 1))
    s = s.numpy().transpose((0, 2, 3, 1))          
    for i in range(batch_size):
        print(i, a[i].shape, c[i].shape, s[i].shape, o)
        ax = plt.subplot(batch_size,3,3*i+1)
        ax.axis('off')
        plt.imshow(a[i])
        ax = plt.subplot(batch_size,3,3*i+2)
        ax.axis('off')
        plt.imshow(c[i])
        ax = plt.subplot(batch_size,3,3*i+3)
        ax.axis('off')
        plt.imshow(s[i])
    plt.tight_layout()
    plt.show()

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