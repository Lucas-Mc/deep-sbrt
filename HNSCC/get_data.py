import os
import math

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class SurvivalDataset(Dataset):
    def __init__(self):
        # Load the data file
        TREATMENT_FILE = 'Patient-and-Treatment-Characteristics.xls'
        treatment_data = pd.read_excel(TREATMENT_FILE)
        # Get the valid dates and patients
        valid_dates = self.get_dates()
        valid_patients = [d.split(os.sep)[1] for d in valid_dates]
        # Assign class attributes
        treatment_data = treatment_data[treatment_data['TCIA code'].isin(valid_patients)]
        # Convert TCIA codes to indices
        self.all_patients = treatment_data['TCIA code'].tolist()
        self.patient = torch.tensor(list(range(len(self.all_patients))))
        self.outcome = torch.tensor([0 if o=='Dead' else 1 for o in treatment_data['Alive or Dead'].tolist()])
        self.num_patients = treatment_data.shape[0]

    def __getitem__(self, index):
        return self.patient[index], self.outcome[index]

    def __len__(self):
        return self.num_patients

    def get_dates(self):
        valid_dates = []
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
                valid_dates.append(path)
        return valid_dates

if __name__ == '__main__':
    dataset = SurvivalDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)
    print(dataiter.next())

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