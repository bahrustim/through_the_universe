#Один из моих проектов на Kaggle
#Изначально он был в формате Jupiter Notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
import os
from keras import models
from keras import layers
from keras.utils import np_utils
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print("Loading train data from CSV .....")
df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print ("Train data loaded")

print ("Processing train data to chunks")
def process_chunk(chunk, rows, drop_col):
    if drop_col ==True:
        chunk = chunk.drop(columns='time_to_failure')
    np_chunk = np.absolute(chunk.values)#.astype('int16')
    np_chunk_means = np_chunk.reshape(-1,rows).mean(1).reshape(1,-1).transpose()
    np_chunk_stds = np_chunk.reshape(-1,rows).std(1).reshape(1,-1).transpose()
    np_chunk_medians = np.median(np_chunk.reshape(-1,rows), 1).reshape(1,-1).transpose()
    np_chunk_averages = np.average(np_chunk.reshape(-1,rows), 1).reshape(1,-1).transpose()
    np_chunk_amaxs = np.amax(np_chunk.reshape(-1,rows), 1).reshape(1,-1).transpose()
    #np_chunk_sums = np.sum(np_chunk.transpose().reshape(-1,rows), 1).reshape(1,-1).transpose()
    return np.concatenate((np_chunk_means, np_chunk_stds, np_chunk_medians, np_chunk_averages, np_chunk_amaxs), 1)

def to_chunks(test_split, ms):
    
    ttf_train = []
    ttf_test = []
    csv_size = len(df.index)
    test_size = csv_size//(1/test_split)
    train_size = csv_size - test_size
    for m in range(ms*2):
        if m < ms:            
            imax = test_size//150000-1            
        if m>= ms:
            imax = train_size//150000-1            
        imax = int(imax)
        for i in range(imax):            
            start=time.time()
            if m < ms:            
                row_start=i*150000+m*150000//ms
            if m>= ms:            
                row_start=test_size+i*150000+(m-ms)*150000//(ms)
            row_start= int(row_start)
            chunk = df.iloc[ row_start:row_start+150000, : ]
            if m < ms:
                ttf_test.append(chunk['time_to_failure'].values[::3000])#.reshape(1, 100))
            if m>= ms:
                ttf_train.append(chunk['time_to_failure'].values[::3000])#.reshape(1, 100))            
            processed_chunk = process_chunk(chunk, 300, True)
            processed_chunk = processed_chunk.reshape(1, processed_chunk.shape[0], processed_chunk.shape[1])

            if i==0:
                collector = processed_chunk
            elif (i+1)%500 == 0 or i == imax-1:
                collector=np.concatenate((collector, processed_chunk), 0)
                if i<501 and m==0:
                    x_test=collector
                elif m<ms:
                    x_test=np.concatenate((x_test, collector), 0)
                
                if i<501 and m==ms:
                    x_train=collector
                elif m>=ms:
                    x_train=np.concatenate((x_train, collector), 0)
                
                print("i: ", i, "|  m: ", m, "|   execution time: ", ex_time)
            elif i%500 == 0:
                collector = processed_chunk
            else:
                collector=np.concatenate((collector, processed_chunk), 0)
            end=time.time()
            ex_time=end-start            
            #, "\nx_train_shape:", x_train.shape, "|  labels_len", len(ttf))
    ttf_test = np.asarray(ttf_test)
    ttf_train = np.asarray(ttf_train)
    return x_test, ttf_test, x_train, ttf_train
x_test, ttf_test, x_train, ttf_train = to_chunks(0.1 , 200)

model = models.Sequential()
model.add(layers.Conv1D(16, 4, activation = 'relu', input_shape = (500, 5)))
model.add(layers.Conv1D(32, 4, activation = 'relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(64, 4, activation = 'relu'))
model.add(layers.Conv1D(128, 4, activation = 'relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(256, 4, activation = 'relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(512, 4, activation = 'relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation = 'relu'))
model.add(layers.Dense(1024, activation = 'relu'))
model.add(layers.Dense(50))
model.summary()

model.compile(optimizer='adam', loss='mae')
model.fit(x_train, ttf_train, epochs = 5, batch_size=64, shuffle = True, validation_split = 0.1)
print("Test data loss:")
model.evaluate(x_test, ttf_test)

segments_list = os.listdir('../input/test')
segments_list.sort()
predictions=[]
for file_name in segments_list:
    file = pd.read_csv('../input/test/'+file_name)
    chunk = file.iloc[:, :]
    processed_chunk = process_chunk(chunk, 300, False)
    processed_chunk = processed_chunk.reshape(1, processed_chunk.shape[0], processed_chunk.shape[1])
    predictions.append(model.predict(processed_chunk)[0][49])
if predictions_global.any() == None:
    predictions_global = np.asarray(predictions).reshape(1,-1)
else:
    predictions_global = np.concatenate((np.asarray(predictions).reshape(1,-1), predictions_global), 0)    
print(predictions_global.shape)

predictions_global = np.median(predictions_global, 0)
predictions_global = predictions_global.tolist()
len(predictions_global)

to_submission = pd.DataFrame({'seg_id':[], 'time_to_failure':[]})
to_submission.seg_id = segments_list
to_submission.time_to_failure = predictions_global
to_submission.seg_id = to_submission.seg_id.str.replace('.csv','', regex=False)
#to_submission.to_csv('to_submission.csv', index=False)

# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
create_download_link(to_submission)
# create a link to download the dataframe
##for i in range(1):
  #  create_download_link(pd.DataFrame(x_train[:, :,i]), "Download {} file".format(i), "xtrain{}".format(i))
    
