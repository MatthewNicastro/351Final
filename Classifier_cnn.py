import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np

train = pd.read_csv('data/sample/training/train_train_sample_pca_cleaned.csv').drop(columns=['Unnamed: 0'])
test = pd.read_csv('data/sample/test/train_test_sample_pca_cleaned.csv').drop(columns=['Unnamed: 0'])

from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, pd_data): 
        self.data = pd_data
    
    def __len__(self):
        return len(self.data.index)
    
    def __getitem__(self, idx): 
        item = self.data.iloc[idx]
        qid = item.qid
        question = item.question_text
        question = question.replace(']','')
        question = question.replace('[','')
        question = question.replace('\r\n', '')
        question = question.split()
        question = np.array([np.float(dim) for dim in question])
        target = item.target
        return (qid, question, target)

train = dataset(train)
test = dataset(test)

batch_size = 64
train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)

class cnn(nn.Module): 
    def __init__(self, 
                 input_size, #size of in vector 
                 num_channels, #num channels
                 kernel_size, #size of kernel
                 pool_size, #size of max pooling kernel
                 ):
        
        super(cnn, self).__init__()
        self.conv = nn.Conv1d(1, num_channels, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.output_size = int((input_size - int(kernel_size/2)*2)/pool_size)
        self.num_channels = num_channels
        self.layer = nn.Linear(self.output_size*self.num_channels, 2)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, inpt): 
        inpt = self.pool(self.relu(self.conv(inpt)))
        inpt = inpt.view(-1, self.output_size*self.num_channels)
        inpt = self.layer(inpt)
        inpt = self.softmax(inpt)
        return inpt



ngpu = 1
device = torch.device("cuda: 0" if(torch.cuda.is_available() and ngpu >= 1) else "cpu")

input_size = 100
num_channels = 3
kernel_size = 3
pool_size = 3

epoch = 30

test_cnn = cnn(input_size, num_channels, kernel_size, pool_size).cuda()
lr = 0.01
criterion = nn.NLLLoss()
optimizer = optim.SGD(test_cnn.parameters(), lr = lr, momentum = 0.9)

def train_model(batch):
    optimizer.zero_grad()
    test_cnn.zero_grad()
    loss = 0
    
    if batch[1].shape[0] == batch_size:
        gpu = batch[1].view(batch_size,1,input_size).to(device).float()
    else:
        gpu = batch[1].view(batch[1].shape[0],1,input_size).to(device).float()
    target = batch[2].to(device)

    predicted = test_cnn(gpu)
    loss = criterion(predicted, target)
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(epoch):
    print("Epoch -> {}".format(i))
    count = 1
    total_error = 0
    for batch in train_loader:
        loss = train_model(batch)
        total_error += loss
        if count % 100 == 0: 
            print('loss -> {}'.format(loss))
        count += 1
    print("Total Batch error={}".format(total_error))

def acc():
    accuracy = 0
    num_batches = 0
    for batch in test_loader:
        if batch[1].shape[0] == batch_size:
            gpu = batch[1].view(batch_size,1,input_size).to(device).float()
        else:
            gpu = batch[1].view(batch[1].shape[0],1,input_size).to(device).float()
        preds = test_cnn(gpu)
        target = batch[2].numpy()
        preds = preds.cpu().detach().numpy()
        preds = np.array([np.argmax(row) for row in preds])
        total_correct = sum(target == preds)
        accuracy += total_correct
        num_batches += 1
    print(accuracy / (num_batches * batch_size))

acc()
    
with open('preds.txt', mode = 'w') as f: 
    f.write('pred = [')
    count = 0
    written = 0
    #replace data -> pd dataframe
    for ind in data.index: 
        if count < 40:
            f.write('{},'.format(data.prediction.iloc[ind]))
            count += 1
            written += 1
        else: 
            f.write('\n')
            f.write('{},'.format(data.prediction.iloc[ind]))
            count = 0
            written += 1
    f.write(']')