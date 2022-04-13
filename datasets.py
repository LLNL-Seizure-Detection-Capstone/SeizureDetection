import pandas as pd
from torch import tensor, reshape
from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np

class CHBMITDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df.insert(loc=23, column='Padding', value=0) # add a column of padding to dataframe

        # Min Max Standard
        normal_df = (self.df-self.df.min()) / (self.df.max()-self.df.min())

        normal_df['Outcome'] = self.df['Outcome']
        normal_df['Padding'] = self.df['Padding']
        self.df = normal_df
    
    def __len__(self):
        return len(self.df)//256 - 1

    def __getitem__(self, index):
        startRow = index * 256
        endRow = startRow + 512 
        # If there is any seizure in the time stamp then label it as a seizure
        outcome = 1.0 if 1.0 in set(self.df.iloc[startRow:endRow, 24]) else 0
        eeg_tensor = tensor( np.array( self.df.iloc[startRow:endRow, 0:24] ))
        eeg_tensor = reshape( eeg_tensor, (1, 512, 24) )
        label_tensor = tensor(outcome)
        return eeg_tensor, label_tensor