#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super (CharDecoder,self).__init__()
        # target_vocab.char2id<-dictionary of the characters in target vocabublary
        V_char=len(target_vocab.char2id)
        self.charDecoder=nn.LSTM(char_embedding_size,hidden_size)
        
        self.char_output_projection=nn.Linear(hidden_size,V_char)
        self. decoderCharEmb=nn.Embedding(num_embeddings=V_char,
                                          embedding_dim=char_embedding_size,
                                          padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab=target_vocab
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #Embedding matrix of character embeddings
        char_embedding = self.decoderCharEmb(input) #len, batch, char_embed_size
        #LSTM decoding (input, (h_0, c_0))->(output, (h_n, c_n))
        #num_layers * num_directions=1
        hidden,dec_hidden=self.charDecoder(char_embedding,dec_hidden)
        # h_ts  shape (len, b, hidden_size)
        #dec_hidden shape (1, batch, hidden_size)
        s_t=self.char_output_projection(hidden)
        #s_t shape (len,batch,v_char)
        return s_t,dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        #- char_sequence corresponds to the sequence x_1 ... x_{n+1} 
        #from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input=char_sequence[:-1] # del x_{n+1}
        # project score, hidden, cell
        s_t,dec_hidden=self.forward(input,dec_hidden)
        
        """
        softmax+loss:
        CrossEntropyLoss(
            weight: Optional[Tensor]=None, 
            size_average=None, 
            ignore_index: int=-100, 
            reduce=None, reduction: str='mean')
        combines nn.LogSoftmax() and nn.NLLLoss() 
        in one single class.
        ignore_index (int, optional): 
            Specifies a target value that is ignored

        Shape:
        Input: input has to be a Tensor of size either (minibatch,C) 
            (N,C) where C = number of classes, or (N,C,d1,d2,...,dK) with K≥1 in the case of K-dimensional loss.

        Target: (N) 

        Output: scalar. if reduction is 'none', then the same size as the target (N)
        """
        #transpose to 1 dim with len=length with contiguous memory
        target=char_sequence[1:].contiguous().view(-1)
        s_t  = s_t.view(-1, s_t.shape[-1])
        loss=nn.CrossEntropyLoss(size_average=False,
                                 ignore_index=self.target_vocab.char2id['<pad>'])
        output=loss(s_t,target)
        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        dec_hidden=initialStates
        start_index=self.target_vocab.start_of_word
        end_index=self.target_vocab.end_of_word
        end_word=self.target_vocab.id2char[end_index]
        # initialStates shape: (1, batch, hidden_size)
        batch_size=initialStates[0].shape[1]
        words=[[-1] * max_length for _ in range(batch_size)]
        words_string=['']*batch_size
        #print(words_string)
        #use torch.tensor to turn a list of character indices into a tensor.
        current_char=torch.tensor([[start_index ]*batch_size], device=device)
        for t in range(max_length):
            score, dec_hidden = self.forward(current_char, dec_hidden) 
            # score shape (1, batchsize, Vocabsize)
            current_char = score.argmax(dim=2) #max on vocab
            # (1, batchsize)
            
            
            #words+=current_char.tolist()
            #print(current_char)
            #convert to list squeeze dim 0 only
            cil = current_char.squeeze(0) .tolist()
            #print(cil)
            for b in range(batch_size):
                words[b][t] =cil[b]
        #print(current_char.shape)
        #print(words)
        #covert to string
        #joining each inner list into a single string
        words_string=[''.join(self.target_vocab.id2char[c] for c in w) for w in words]
        #print(words_string)
        #words_tensor=torch.tensor(words, device=device)
        split_w=map(lambda w: w.split(end_word)[0], words_string)
       # print(words_tensor)
        output=list(split_w)
        #print(output)
        return(output)
        ### END YOUR CODE

