import torch
import torch.nn as nn
import math

class tokenizer(nn.Module):
    #word level tokenizer : each space defines boundary of word and each word is mapped to a number 
    def __init__(self, ) -> None:
        super().__init__(*args, **kwargs)

class inputEmbedding(nn.Module):

    #Embedding step is mainly to convert the words into 512 dim vectors and tensor does it for us

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #Simply maps the numbers and vector of dimension 512
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  #we mult by this number because its given in the paper to achieve better results

class positionalEncoding(nn.Module):  

    #seq_len is the max length of the sentence 
    #dropout is to make model less overfitting
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        #Creating a matrix of dim(seq_length x d_model)
        
        pe_mat = torch.zeros(seq_length, d_model)
        
        #Formula given in paper :
        #   PE(pos, 2i|2i+1) = sin|cos(pos/(10000^(2i/d_model)))
    
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        #Creates tensor of shape (seq_length, 1)            
        deno = torch.exp(torch.arange(0,d_model, 2).float() * (-math.log(10000)/d_model))
        
        pe_mat[:, 0::2] = torch.sin(pos*deno)
        pe_mat[:, 1::2] = torch.cos(pos*deno)
        
        pe_mat = pe_mat.unsqueeze(0)
        #makes tensor of shape (1,seq_len, d_model)
        
        self.register_buffer('pe_mat', pe_mat)
    
    def forward(self, x):
        x = x + (self.pe_mat[:,:x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
        
class layerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        
        #nn.Parameter makes the parameter learnable
        self.alpha = nn.Parameter(torch.ones(1))   #multiplied
        self.bias = nn.Parameter(torch.zeros(1))   #added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        
        return self.alpha * (x-mean)/(std + self.eps) + self.bias
    
class feedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self,x):
        #INP : (Batch, seq_len, d_model)  --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class multiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model=d_model
        self.h=h

        self.dk = d_model//h        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_res = nn.Linear(d_model, d_model)
        
        self.dropout= nn.Dropout(dropout)
    
    @staticmethod #so that it can be used wihout an instance of multihead
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dk = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
        
    def forward(self, q,k,v, mask):
        
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
                
        #(Batch, seq_len, d_model) --> (Batch, seq_len, h, dk) --Trans--> (Batch, h, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1,2)
        
        x, self.attention_scores= multiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        #(Batch, h, seq_len, dk) --> (Batch, seq_len, h, dk) --> (Batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.dk)
        
        return (self.w_res(x))
        
class residualCon(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class linearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.lin(x), dim = -1)


        
