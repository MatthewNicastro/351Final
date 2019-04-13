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

class bidirec_lstm_1dcnn(nn.Module): 
    def __init__(self, 
                 input_size, #size of in vector
                 hidden_size,
                 num_layers,
                 num_channels, #num channels
                 kernel_size, #size of kernel
                 pool_size, #size of max pooling kernel
                 dropout,
                 ID, 
                 lr, 
                 optimizer
                 ):
        super(bidirec_lstm_1dcnn, self).__init__()
        self.input_size = input_size
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout = dropout
        self.lr = lr
        
        self.ID = ID
        self.f1 = 0
        
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
        self.soft = nn.LogSoftmax(dim = 1)
    
    def getParams(self):
        return self.hidden_size, self.num_layers, self.num_channels,self.kernel_size,self.pool_size, self.dropout, self.lr
    
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
        inpt = self.soft(inpt)
        return inpt

class genetic_training(object): 
    def __init__(self, population_size, device, optimizer, criterion):
        self.population_size = population_size
        self.population = []
        input_size = 300
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        for i in self.population_size:     
            hidden_size = np.random.randint(100, 800)
            num_layers = np.random.randint(1, 6)
            num_channels = np.random.randint(2, 300)
            kernel_size = np.random.randint(1, hidden_size)
            pool_size = np.random.randint(2, int(hidden_size/2))
            dropout = np.random.rand()/1.5
            lr = np.random.rand()/10
            try:
                self.population.append(bidirec_lstm_1dcnn(input_size, 
                                                          hidden_size, 
                                                          num_layers,
                                                          num_channels,
                                                          kernel_size,
                                                          pool_size,
                                                          dropout,
                                                          i,
                                                          lr
                                                          ).cuda())
            except:
                self.population.append(bidirec_lstm_1dcnn(input_size, 
                                                          hidden_size, 
                                                          num_layers,
                                                          num_channels,
                                                          kernel_size,
                                                          pool_size,
                                                          dropout,
                                                          i,
                                                          lr).cuda())
    
    def kfolds(self, fold_list): 
        val_ind = np.random.randint(10)
        train = []
        np.random.shuffle(fold_list)
        for i in range(10): 
            if i != val_ind: 
                train.append(fold_list[i])
            else:
                val = fold_list[i]
        return train, val
    
    def validate(self, loader, model):
        act = np.array([])
        pred = np.array([])
        for batch in loader:
            gpu = batch.question_text.to(self.device).long()
            preds = model(gpu)
            target = batch.target.numpy()
            preds = preds.cpu().detach().numpy()
            preds = np.array([np.argmax(row) for row in preds])
            act = np.concatenate((act, target))
            pred = np.concatenate((pred, preds))
        formula1 = f1(act, pred)
        print(model.ID,'val f1 ->',formula1)
        return formula1
    
    def fit_population(self, epoch, fold_loader): 
        for model in self.population:
            print('Training {}'.format(model.id))
            for i in range(epoch):
                print('Epoch {}'.format(i))
                train_folds, val_fold = self.kfolds(fold_loader)
                
                for fold in train_folds:
                    for batch in fold:
                        self.optimizer.zero_grad()
                        model.zero_grad()
                        loss = 0
                        
                        gpu = batch.question_text.to(self.device).long()
                        target = batch.targetto(self.device).long()
                    
                        predicted = model(gpu)
                        loss = self.criterion(predicted, target)
                        loss.backward()
                        self.optimizer.step()
            model.f1 = self.validate(val_fold, model)
    
    def breed(self, father, mother, k): 
        def mutation(gen, alpha):
            if gen < 5: 
                mut = int(np.random.rand()*alpha)
            else:
                mut = np.random.rand()/alpha
            return mut
        genome = np.random.rand(7) > k
        alpha = int(1/abs(father.f1 - mother.f1))
        father_code = father.getParams()
        mother_code = mother.getParams()
        params = []
        i = 0
        for gene in genome: 
            if gene:
                params.append(father_code[i] + mutation(i, alpha))
            else: 
                params.append(mother_code[i]+ mutation(i, alpha))
            i += 1
        return bidirec_lstm_1dcnn(*params)
    
    def generateOffspring(self): 
        self.population.sort(key = lambda x: x.f1)
        next_gen = []
        breeding_season = True
        while breeding_season:
            for i in range(self.population_size-2):
                for j in range(i+1, self.population_size):
                    if self.population[i].f1 - self.population[j].f1 > np.random.rand():
                        next_gen.append(self.breed(self.population[i], self.population[j]))
            if len(next_gen) >= self.population_size: 
                breeding_season = False
        self.population = next_gen
        self.population_size = len(next_gen)
        
        
        
        
        
        
        
        
        