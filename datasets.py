import pandas as pd
from torch import tensor, reshape
from torch.utils.data import Dataset
from sklearn import preprocessing
import numpy as np
import mne
import os

class CHBMITDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df.insert(loc=23, column='Padding', value=0) # add a column of padding to dataframe
        
        # Normalize & Standardize data
        # Mean Normal
        #normal_df = (self.df-self.df.mean()) / self.df.std()

        # Min Max Standard
        normal_df = (self.df-self.df.min()) / (self.df.max()-self.df.min())

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

class RawCHBMITDataset(Dataset):
    def __init__(self, edf_folder, transform=None):
        self.df = pd.DataFrame()
        for filename in os.listdir(edf_folder):
            filename = os.path.join(edf_folder, filename)
            print("filepath: ", filename)
            data = mne.io.read_raw_edf(filename)
            raw_data = data.get_data()
            raw_df = pd.DataFrame(raw_data)
            self.df = pd.concat([self.df, raw_df], axis=1)
        
        
        # Normalize & Standardize data
        # Mean Normal
        #normal_df = (self.df-self.df.mean()) / self.df.std()

        # Min Max Standard
        self.df = (self.df-self.df.min()) / (self.df.max()-self.df.min())
        print("dataframe: ", self.df)

        # normal_df['Outcome'] = self.df['Outcome']
        # normal_df['Padding'] = self.df['Padding']
        # self.df = normal_df

        # We may need to normalize somehow by batch
    
    def __len__(self):
        return (len(self.df.columns))//256 - 1 # 14745600 / 256 - 1 = 57600 -> number of seconds of data that we have

    def __getitem__(self, index):
        startCol = index * 256
        endCol = startCol + 256

        eeg_tensor = tensor( np.array( self.df.iloc[0:23, startCol:endCol] ))
        eeg_tensor = reshape( eeg_tensor, (1, 23, 256) )
        return eeg_tensor 
        
