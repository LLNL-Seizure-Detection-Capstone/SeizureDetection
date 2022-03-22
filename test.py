import mne
import pandas as pd

file = "./chb01/chb01_02.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
print("raw_data:\n", raw_data)
print("row size:", len(raw_data))
df = pd.DataFrame(raw_data)
print("col size:", len(df.columns))
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names

print(info)