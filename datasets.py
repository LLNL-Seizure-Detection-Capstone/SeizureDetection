import pandas as pd
from torch import tensor
from torch.utils.data import Dataset
from sklearn import preprocessing

class CHBMITDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df.insert(loc=23, column='padding', value=0) # add a column of padding to dataframe
        
        # Normalize & Standardize data
        # self.df_min_max_scale = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(self.df)) # range [0, 1]
        # self.df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(self.df_min_max_scale)) # centers values around 0 with std dev of 1
       
        print("seizure starts ", self.df.iloc[8190]) # this row should have an outcome of 1 but instead is printing outcome 0?
        self.transform = transform
    
    def __len__(self):
        return len(self.df)//256 - 1 # num of rows / 256 - 1 = 8190

    def __getitem__(self, index):
        startRow = index * 256
        endRow = startRow + 512 + 1 # changing this value doesn't affect anything (it still returns 510 rows???)

        outcome = 1 if 1.0 in self.df['Outcome'] else 0 # if there is at least a single 1 in any of the rows return 1 else 0
        #outcome = self.df.Outcome.mode()
        return tensor(self.df.iloc[startRow:endRow, 0:24]), tensor(outcome) # if you change 24 to 25 you will see outcome column
        
        # locate time stamp
        # read in that edf file
        # set y_label = (0 or 1 depending on seizure)

        # if self.transform:
        #     file = self.transform(file)
        # return (file, y_label)

        #return a dataframe 512 x 24 with the last column padded with zeroes and the y label