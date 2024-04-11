import argparse
import torch
import torch.nn as nn
import numpy as np
import re
import spacy
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import json
import time
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import wikipedia
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

kUNK = '<unk>'
kPAD = '<pad>'
kSEP = '<sep>'
# You don't need to change this funtion
def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            #ans=ans.replace('_',' ')
            for j in ans.split('_'):
               class_to_i[j] = i
               i_to_class[i] = j
               i+=1
    return class_to_i, i_to_class

# You don't need to change this funtion


def load_glove_vectors(glove_file):
    glove_vectors = {}
    word2idx = {}
    idx = 0

    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_vectors[word] = vector
            word2idx[word] = idx
            idx += 1

    return glove_vectors, word2idx



def load_data(filename, lim):
    """
    load the json file into data list
    """

    data=list()
    with open(filename) as json_data:
        c=0
        if lim>0:
            questions = json.load(json_data)["questions"][:lim]
        else:
            questions = json.load(json_data)["questions"]
        for q in questions:
            
            l= ["10","For 10 points","for 10 points","For 10 point","10 point","For ten points","FTP","For 10"]
            for i in l:
              try:
                cont,ques = q['text'].split(i)
                break
              except:
                continue
            
            q_cont = nltk.word_tokenize(cont)
            q_ques = nltk.word_tokenize(ques)
            q_text = q_cont + list(kSEP) + q_ques
            #label = q['category']
            label = q['page']
            label = list(label.replace('_',' '))
            #print(label)
            data.append((q_text,label[0]))
            
    #print(data)
    return data

# You don't need to change this funtion
def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {kPAD: 0, kUNK: 1 , kSEP: 2}
    ind2word = {0: kPAD, 1: kUNK, 2: kSEP}
    for q_text, a in exs:
        for w in q_text:
            words.add(w)
        try:
           s = a.split("_")
           for i in s:
              words.add(i)
        except:
           words.add(a)
        
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word


class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###You don't need to change this funtion
    def __init__(self, examples, word2ind, num_classes, class2ind=None):
        self.questions = []
        self.labels = []
        
        for qq, ll in examples:
            self.questions.append(qq)
            self.labels.append(ll)
        
        if type(self.labels[0])==str:
            for i in range(len(self.labels)):
                try:
                    self.labels[i] = class2ind[self.labels[i]]
                except:
                    self.labels[i] = num_classes
        #print(self.labels)
        self.word2ind = word2ind
    
    ###You don't need to change this funtion
    def __getitem__(self, index):
        return self.vectorize(self.questions[index], self.word2ind), \
          self.labels[index]
    
    ###You don't need to change this funtion
    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(ex, word2ind):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        vec_text = [0] * len(ex)

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized    
        ####Your code here
        
        for i in range(0,len(ex)):
          if ex[i] in word2ind.keys():    
             vec_text[i]=word2ind[ex[i]]
          else:
             vec_text[i]=word2ind[kUNK]
        

        return vec_text



    

###You don't need to change this funtion

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 
    Keyword arguments:
    batch: list of outputs from vectorize function
    """
    

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch



def recall_at_k(logits, labels, k):
    # Sort logits and get the top-k indices
    _, top_k_indices = logits.topk(k, dim=1)

    # Convert labels to a tensor
    labels_tensor = torch.LongTensor(labels)

    # Check if the true label is in the top-k indices
    correct = torch.sum(top_k_indices == labels_tensor.view(-1, 1))

    # Calculate recall@k
    recall = correct.item() / len(labels)
    
    return recall

'''def evaluate(data_loader, model, device, top_k=5):
    model.eval()
    num_examples = 0
    correct_predictions = 0

    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        # Forward pass to get logits
        logits = model(question_text, question_len, is_prob=True)

        # Get the top-k predicted classes
        top_k_predictions = logits.topk(top_k, dim=1)[1]

        # Iterate through the batch
        for i in range(len(top_k_predictions)):
            # Check if any of the top-k predicted classes match the actual labels
            if any(prediction in labels[i] for prediction in top_k_predictions[i]):
                correct_predictions += 1

        num_examples += question_text.size(0)

    accuracy = correct_predictions / num_examples
    print(f'Top-{top_k} Accuracy:', accuracy)
    return accuracy'''





def evaluate(data_loader, model, device, k=5):
    model.eval()
    num_examples = 0
    recall_sum = 0

    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        logits = model(question_text, question_len, is_prob=True)

        batch_recall = recall_at_k(logits, labels, k)
        recall_sum += batch_recall
        num_examples += question_text.size(0)

    avg_recall = 100 * (recall_sum / num_examples)
    print(f'recall@{k}', avg_recall)
    return avg_recall







def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        #### Your code here
        logits = model(question_text, question_len,is_prob=True)
        #print(labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        clip_grad_norm_(model.parameters(), args.grad_clipping) 
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()
        #print(print_loss_total,epoch_loss_total)

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
        
    return accuracy





def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    weights_tensor = torch.from_numpy(weights_matrix)
    emb_layer.load_state_dict({'weight': weights_tensor})
    if non_trainable:
        emb_layer.weight.requires_grad = True

    return emb_layer, num_embeddings, embedding_dim


class DanModel(nn.Module):
    def __init__(self, n_classes, weights_matrix, n_hidden_units=50, nn_dropout=0.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout

        # Create an embedding layer and initialize it with the GloVe vectors
        self.emb_dim = 50
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        

        # Linear layers for classification
        self.linear1 = nn.Linear(self.emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        self.layers = nn.Sequential(
                  self.linear1,
                  nn.ReLU(),
                  nn.Dropout(self.nn_dropout),
                  self.linear2
         )
        

    def forward(self, input_text,text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text: List of word indices
        is_prob: if True, output the softmax of the last layer
        """
       
        #input_text = input_text.to(torch.int64)
        #max_index = input_text.max().item()
        
        embeddings = self.embedding(input_text)
        avg_embeddings = torch.sum(embeddings, dim=1) / text_len.view(-1, 1).float()
        #avg_embeddings = torch.mean(embeddings, dim=1)
        logits = self.layers(avg_embeddings)

        if is_prob:
            softmax = nn.Softmax(dim=1)
            logits = softmax(logits)

        return logits

# You basically do not need to modify the below code 
# But you may need to add funtions to support error analysis 

def get_glove_vocab_size(glove_file):
    vocab_size = 0
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            vocab_size += 1
    return vocab_size


def find_misclassify(data_loader, question_set, model, device):
    """
    finds misclassified examples in the dev set

    Keyword arguments:
    question_set: the dev set
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        logits = model.forward(question_text,question_len)

        top_n, top_i = logits.topk(1)

        # Note that since our batch size is 1, the error tells whether the example is classified correctly
        error = torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
        if error != 0:
            print("prediction:", top_i.squeeze())
            print("actual:",torch.LongTensor(labels))
            print(question_set.examples[idx])
            print(question_set[idx])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--train-file', type=str, default='data/qanta.train.json')
    parser.add_argument('--dev-file', type=str, default='data/qanta.dev.json')
    parser.add_argument('--test-file', type=str, default='data/qanta.test.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    ### Load data
    train_exs = load_data(args.train_file, args.limit)
    print("training done")
    dev_exs = load_data(args.dev_file, -1)
    print("devops done")
    #test_exs = load_data(args.test_file, -1)

    vectors,w2ind = load_glove_vectors('glove.6B.50d.txt')
    
    voc, word2ind, ind2word = load_words(train_exs)

    #get num_classes from training + dev examples - this can then also be used as int value for those test class labels not seen in training+dev.
    num_classes = len(list(set([ex[1] for ex in train_exs+dev_exs])))
    print(num_classes)

    #get class to int mapping
    class2ind, ind2class = class_labels(train_exs + dev_exs)  
    vocab_size = get_glove_vocab_size('glove.6B.50d.txt')

    matrix_len = len(ind2word)
    weights_matrix = np.zeros((matrix_len, 50))
    print(weights_matrix.shape)
    words_found = 0
    emb_dim=50
    for i, word in enumerate(ind2word):
      try: 
        weights_matrix[i] = vectors[word]
        words_found += 1
      except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

    if args.test:
        model = torch.load(args.load_model)
        #### Load batchifed dataset
        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, voc)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)
    else:
        if args.resume:
            model = torch.load(args.load_model)
        else:
            model = DanModel(len(voc), weights_matrix)
            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = QuestionDataset(train_exs, word2ind, num_classes, class2ind)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler, num_workers=0,
                                               collate_fn=batchify)
        accuracy=0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=0,
                                               collate_fn=batchify)
            accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
        print('start testing:\n')

        test_dataset = QuestionDataset(test_exs,word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)