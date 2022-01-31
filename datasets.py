import pandas as pd
from torch import tensor, reshape
from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np

class CHBMITDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df.insert(loc=23, column='Padding', value=0) # add a column of padding to dataframe
        
        # Normalize & Standardize data
        normal_df = (self.df-self.df.mean()) / self.df.std()
        normal_df['Outcome'] = self.df['Outcome']
        normal_df['Padding'] = self.df['Padding']
        self.df = normal_df

        # We may need to normalize somehow by batch
    
    def __len__(self):
        return len(self.df)//256 - 1 # num of rows / 256 - 1 = 8190

    def __getitem__(self, index):
        startRow = index * 256
        endRow = startRow + 512 # changing this value doesn't affect anything (it still returns 510 rows???)

        outcome = 1.0 if 1.0 in set(self.df.iloc[startRow:endRow, 24]) else 0 # if there is at least a single 1 in any of the rows return 1 else 0
        #outcome = self.df.Outcome.mode()
        eeg_tensor = tensor( np.array( self.df.iloc[startRow:endRow, 0:24] ))
        eeg_tensor = reshape( eeg_tensor, (1, 512, 24) )
        label_tensor = tensor(outcome)
        return eeg_tensor, label_tensor # if you change 24 to 25 you will see outcome column
        
        # locate time stamp
        # read in that edf file
        # set y_label = (0 or 1 depending on seizure)

        # if self.transform:
        #     file = self.transform(file)
        # return (file, y_label)

        #return a dataframe 512 x 24 with the last column padded with zeroes and the y label