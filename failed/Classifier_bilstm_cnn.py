import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext
from torchtext.data import Iterator, BucketIterator
from torchvision import transforms, utils
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score as f1

#batch_size for training
batch_size = 200
verbose = 100

#Training on the gpu
ngpu = 1
device = torch.device("cuda: 0" if(torch.cuda.is_available() and ngpu >= 1) else "cpu")

#Model parameters
input_size = 300
hidden_dim = 500
num_layers = 2
num_channels = 10
kernel_size = 10
pool_size = 10
dropout = 0.5
lr = 0.01
criterion = nn.NLLLoss()

#Evaluation metrics for model
accuracy_train = []
f1_train = []
accuracy_val = []
f1_val = []

total_batch_error = []

#Number of epochs
epoch = 10

def split_text(text):
    return text.split(' ')

def prepare_sequences():
    tokenizer = split_text # the function above is the function we will be using to tokenize the text
    TEXT = torchtext.data.ReversibleField(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False) # sequential and use_vocab=False since no text (binary)
    QID = torchtext.data.Field(sequential=False, use_vocab=False)
    train_datafields = [('', None),("question_text", TEXT), ("target", LABEL)]
    train = torchtext.data.TabularDataset( # If we had a validation set as well, we would add an additional .splits(...)
            path="data/use_this/train.csv", # the root directory where the data lies
            format='csv',
            # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
            skip_header=True, 
            fields=train_datafields)
         
    val_datafields = [('', None),
                     ("question_text", TEXT), ("target", LABEL)] 
    val = torchtext.data.TabularDataset( 
                path="data/use_this/val.csv",
                format="csv",
                skip_header=True,
                fields=val_datafields
    )
    test_datafields = [("qid", QID),('', None),
                       ("question_text", TEXT)]
    test = torchtext.data.TabularDataset( 
                path="data/use_this/test.csv",
                format="csv",
                skip_header=True,
                fields=test_datafields
    )
    return TEXT, LABEL, train,  val, test

TEXT, LABEL, train, val, test = prepare_sequences()
TEXT.build_vocab(train, val, test, vectors="glove.6B.300d")

train_loader = BucketIterator(
     train, # we pass in the datasets we want the iterator to draw data from
     batch_size=batch_size,
     shuffle = True,
     sort_key=lambda x: len(x.question_text), # the BucketIterator needs to be told what function it should use to group the data.
     sort_within_batch=False, # sorting would add bias
     repeat=False 
)

val_loader = Iterator(
        val,
        batch_size=batch_size,
        sort=False,
        shuffle = True,
        sort_within_batch=False,
        repeat=False
)

test_loader = Iterator(
        test,
        batch_size=batch_size,
        sort=False,
        sort_within_batch=False,
        repeat=False
)

class bidirec_lstm_1dcnn(nn.Module): 
    def __init__(self, 
                 input_size, #size of in vector
                 hidden_size,
                 num_layers,
                 num_channels, #num channels
                 kernel_size, #size of kernel
                 pool_size, #size of max pooling kernel
                 dropout
                 ):
        super(bidirec_lstm_1dcnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        
        self.emb = nn.Embedding(len(TEXT.vocab), self.input_size)
        
        self.lstm = nn.LSTM(self.input_size, 
                            self.hidden_size, 
                            num_layers = self.num_layers,
                            dropout = self.dropout,
                            bidirectional = True)
        
        self.conv = nn.Conv1d(1, 
                              num_channels,
                              kernel_size)
        
        self.pool = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.output_size = int((self.hidden_size*2 - int(self.kernel_size/2)*2)/self.pool_size) # 2 for bi direction
        self.layer = nn.Linear(self.output_size*self.num_channels, 2)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, inpt): 
        #bid lstm layer
        inpt = self.emb(inpt)
        batch_size = inpt.shape[1]
        inpt,_ = self.lstm(inpt)
        #formating output from lstm to be of id conv size
        inpt = inpt[-1,:,:].view(batch_size, 1,self.hidden_size * 2)

        #1D convolutional layer
        inpt = self.conv(inpt)
        inpt = self.relu(inpt)
        inpt = self.pool(inpt)
        inpt = inpt.view(batch_size, -1)
        inpt = self.layer(inpt)
        inpt = self.softmax(inpt)
        return inpt

try:
    bid_lstm_cnn = bidirec_lstm_1dcnn(input_size, hidden_dim, num_layers, num_channels, kernel_size, pool_size, dropout).cuda()
except:
    bid_lstm_cnn = bidirec_lstm_1dcnn(input_size, hidden_dim, num_layers, num_channels, kernel_size, pool_size, dropout).cuda()

optimizer = optim.SGD(bid_lstm_cnn.parameters(), lr = lr, momentum = 0.7)

def train_model(batch):
    optimizer.zero_grad()
    bid_lstm_cnn.zero_grad()
    loss = 0
    
    gpu = batch.question_text.to(device).long()
    target = batch.target.to(device).to(device).long()

    predicted = bid_lstm_cnn(gpu)
    loss = criterion(predicted, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def acc(loader):
    accuracy = 0
    num_batches = 0
    act = np.array([])
    pred = np.array([])
    for batch in loader:
        gpu = batch.question_text.to(device).long()
        preds = bid_lstm_cnn(gpu)
        target = batch.target.numpy()
        preds = preds.cpu().detach().numpy()
        preds = np.array([np.argmax(row) for row in preds])
        total_correct = sum(target == preds)
        
        act = np.concatenate((act, target))
        pred = np.concatenate((pred, preds))
        
        accuracy += total_correct
        num_batches += 1
    ass = accuracy / (num_batches * batch_size)
    print(ass)
    formula1 = f1(act, pred)
    print(formula1)
    tn,fp,fn,tp = cm(act, pred).ravel()
    print('True positives -> {}\nFalse positives -> {}\nTrue negatives -> {}\nFalse negatives -> {}\n'.format(tp,fp,tn,fn))
    return ass,formula1

def plot():
    plt.plot([i for i in range(len(accuracy_train))], f1_train, label = 'F1 Training set')
    plt.plot([i for i in range(len(accuracy_train))], accuracy_train, label = 'Acc. Training set')
    plt.plot([i for i in range(len(accuracy_val))], f1_val, label = 'F1 Validation set')
    plt.plot([i for i in range(len(accuracy_val))], accuracy_val, label = 'Acc. Validation set')
    plt.legend()
    plt.title('Pre Training vs Pre Cross-validation F1 score/Acc.')
    plt.xlabel('Num. Epoch\'s')
    plt.ylabel('F1 Score/Acc.')
    plt.show()
'''
for i in range(epoch):
    print("Epoch -> {}".format(i))
    count = 1
    total_error = 0
    for batch in train_loader:
        loss = train_model(batch)
        total_error += loss
        if count % verbose == 0: 
            print('loss -> {}'.format(loss))
        count += 1
    print("Total Batch error={}".format(total_error))
    total_batch_error.append(total_error)
    train_set = acc(train_loader)
    accuracy_train.append(train_set[0])
    f1_train.append(train_set[1])
    val_set = acc(val_loader)
    accuracy_val.append(val_set[0])
    f1_val.append(val_set[1])

plot()

torch.save(bid_lstm_cnn.state_dict(), 'bid_lstm_cnn_v6.0_reduced.pt')
''' 
bid_lstm_cnn.load_state_dict(torch.load('bid_lstm_cnn_v6.0.pt'))
bid_lstm_cnn.eval()
predictions = {}
for batch in test_loader:
    qid = batch.qid
    question = batch.question_text.to(device).long()
    preds = bid_lstm_cnn.forward(question)
    preds = preds.cpu().detach().numpy()
    preds = np.array([np.argmax(row) for row in preds])
    for idx, ID in enumerate(qid):
        predictions[ID.item()] = preds[idx]
submission = {'qid' : [], 'prediction' : []}
import pandas as pd
test_qid = pd.read_csv('data/use_this/test.csv')
test_qid.columns = ['indx', 'qid','question']
for indx in test_qid.indx: 
    submission['qid'].append(test_qid.qid[test_qid.indx == indx].item())
    submission['prediction'].append(predictions[indx])
submission_ = pd.DataFrame.from_dict(submission)
print(len(submission_))
submission_.to_csv('submission_BiLSTM_CNN_v6.0.csv', index = False)