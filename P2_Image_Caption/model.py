import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, drop_prob=0.2):
        super(DecoderRNN, self).__init__()
        # initialize parameters 
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        #define layers 
        
        # embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        
        # lstm layer 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        
        # dropout layer to counter overfitting 
        self.dropout = nn.Dropout(drop_prob)
        
        # output from lstm to vocab 
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(1)
        
        # init weights 
        self.init_weights()
    
    def forward(self, features, captions):
        # remove the last <end> and embedding word feature 
        captions = self.word_embedding(captions[:,:-1])            # batch_size x (seq_len - 1) x embed_size
        
        # concatenate the visual embedding and the word embedding 
        features_un = features.unsqueeze(1)              # batch_size x embed_size => # batch_size x 1 x embed_size  
        embed = torch.cat((features_un, captions), dim=1)    # batch_size x seq_len x embed_size 
        
        # pass to LSTM layer 
        lstm_out, _ = self.lstm(embed)                  # batch_size x seq_len x hidden_size
        
        # pass through a drop_out layer 
        output = self.dropout(lstm_out)                 
        
        # from lstm to vocab 
        output = self.hidden2out(output)                 # batch_size x seq_len x vocab_size 
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption = []
        for i in range(max_len):
            lstm_hidden, states = self.lstm(inputs, states)   # lstm_hidden (1, 1, hidden_size )
            lstm_out = self.hidden2out(lstm_hidden.squeeze(1)) # vocab (1, vocab_size)
            _, predicted = lstm_out.max(1)               # predicted (1, 1)
            caption.append(predicted.item())             # add the predicted token to the output caption
            inputs = self.word_embedding(predicted)         # next input caption (1, embed_size) 
            inputs = inputs.unsqueeze(1)                # inputs (1, 1, embed_size) 
        return caption
            
            
    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        
      
        # self.word_embedding.weight.data.uniform_(-0.1, 0.1)
        # self.hidden2out.weight.data.uniform_(-0.1, 0.1)
        
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.word_embedding.weight)
        torch.nn.init.xavier_normal_(self.hidden2out.weight)
        self.hidden2out.bias.data.fill_(0.01)
        
        # init forget gate bias to 1
        # below code taken from https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)    
            
            
            
            
            
            
            
            
            
        
        
        
        