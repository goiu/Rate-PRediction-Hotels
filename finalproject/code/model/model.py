import numpy as np
import pandas as pd
import torch
# from tensorflow import 
import tensorflow as tf
# from tensorflow import LSTM, Linear, Dropout, MaxPool1d, GRU, Conv1d, Embedding, Sequential, ReLU, Softmax, Sigmoid
from preprocessing import preprocess
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


#constants
GPU = False
MAX_WORDS = 50
#Depends on which classification (2, 3 or 5)
NUMBER_OF_CLASSES = 2
#Set large vocab size for embedding matrix
VOCAB_SIZE = 500000
EPOCHS = 50
BATCH_SIZE = 100

class Model(tf.keras.Model):
    def __init__(self, classification):
        super(Model, self).__init__()
        #Define batch size
        self.batch_size = 100
        self.linear_size_one = 300 
        self.linear_size_two = 100 
        
        #Define embedding matrix
        self.embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 128)
        
        #Define layers
        #Type error
        # self.LSTM = tf.keras.layers.LSTM((MAX_WORDS*128, self.linear_size_one))
        self.LSTM = tf.keras.layers.LSTM(self.linear_size_one, return_sequences=True)
        self.l1 = tf.keras.layers.Dense(self.linear_size_two, activation='relu')
        #Pass in the number of output classes
        self.l2 = tf.keras.layers.Dense(classification, activation='softmax')

        # self.sigm = tf.keras.activations.sigmoid()
        self.optimizer = tf.keras.optimizers.Adam()
        #Loss list for plotting the losses
        self.loss_list = []

    def call(self, reviews):
        #The shape of the self.embedding output will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)
        # l1_out = tf.reshape(l1_out, (self.batch_size, MAX_WORDS**128))

        #Pass inputs through LSTM
        l2_out = self.LSTM(l1_out)

        l2_out = tf.reshape(l2_out, (self.batch_size, -1))
        
        #Pass through dense layers
        l3_out = self.l1(l2_out)
       
       
        final_out = self.l2(l3_out) 

        #Use softmax to get probability distribution 
        # final_out = self.softmax(l5_out)

        # print(final_out)
        
        return final_out

    def loss(self, labels, predictions):

        #Convert labels to tensor
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        output = loss(labels, predictions)
        return output


    def accuracy(self, predictions, labels):
        
        correct_count = 0
        count = 0
        #Run through each input in batch
        for i in range(len(predictions)):
            #Helps to indicate which predictions are correct/shows what the model learned for that input
            # print(predictions[i])
            # print(labels[i])
            # print(tf.argmax(predictions[i]).numpy())
            #Returns the indices of the maximum value thus if correctly predicted increments counter
            if tf.argmax(predictions[i]).numpy() == labels[i]:
                correct_count += 1
            count += 1
        
        #Return the correct predictions over total to give accuracy for that batch
        return correct_count/count


    def train_torch(self, inputs, labels):
        #Define optimizer to use in backpropogation
        optimizer = tf.optimizers.Adam(learning_rate=0.005)
        for i in range(int(len(inputs)/self.batch_size)):
            print("Training batch: ", i)

            #Get the next batch of inputs and labels
            input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
            labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]
   
            #Convert inputs to tensor
            input_batch = torch.tensor(input_batch)
           
            #Run forward pass
            probabilites = self.call(input_batch)

            #Calculate the loss
            loss = self.loss(labels_batch, probabilites)
            print("Loss from batch: ", i, "i", loss)
            self.loss_list.append(loss.item())
            
            #Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return 
    


    
    # def test_torch(self, inputs, labels):
    #     accuracy_list = []

    #     for i in range(int(len(inputs)/self.batch_size)):
    #         print("Testing batch: ", i)

    #         #Get next batch of inputs and labels
    #         input_batch = inputs[i*self.batch_size: i*self.batch_size + self.batch_size]
    #         labels_batch = labels[i*self.batch_size: i*self.batch_size + self.batch_size]

    #         #Convert inputs to tensor to run through model
    #         input_batch = torch.tensor(input_batch)

    #         #Get the probability distribution for the batch
    #         probabilites = self.call(input_batch)

    #         #Calculate the accuracy
    #         accuracy = self.accuracy(labels_batch, probabilites)
    #         print("batch accuracy: ", accuracy)
            
    #         #Append accuracy to list
    #         accuracy_list.append(accuracy)
        
    #     #Return the average accuracy across all batches
    #     return np.average(accuracy_list)

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    test_accuracy = []
    num_batches = int(len(test_inputs)/model.batch_size)
    for batch in range(num_batches):
        low = model.batch_size * batch
        high =  np.multiply(model.batch_size, batch) + model.batch_size
        logit = model.call(test_inputs[low:high])
        accuracy = model.accuracy(logit, test_labels[low:high])
        test_accuracy.append(accuracy)
    return_me = np.average(test_accuracy)
    return return_me

def train(model, inputs, labels):
    tl = 0 
    for i in range(int(len(inputs)/model.batch_size)):
        print("Training batch: ", i)
        input_batch = inputs[i*model.batch_size: i*model.batch_size + model.batch_size]
        labels_batch = labels[i*model.batch_size: i*model.batch_size + model.batch_size]
        input_batch = tf.Variable(input_batch)
        with tf.GradientTape() as tape:
            probs = model.call(input_batch)
            loss = model.loss(labels_batch, probs)
            model.loss_list.append(loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model.loss_list




def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    
    """
    #losses = losses.numpy()
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

def main():
    
    print("called main")

    #Instantiate the model
    #Change classification to be number of classes you want model to differentiate between
    model = Model(classification=2)

    #Get the train and test inputs and labels from preprocess
    #Preprocesses labels depending on what type of classification: binary/multi-class
    train_inputs, test_inputs, train_labels, test_labels = preprocess(classification=2)

    #Train the model

    
    #Get the accuracy from testing

    train(model, train_inputs, train_labels)
    accuracy = test(model, test_inputs, test_labels)
    print("accuracy", accuracy)
    visualize_loss(model.loss_list)
    
     
if __name__ == "__main__":
    main()
