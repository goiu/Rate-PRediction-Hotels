import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from preprocessing import preprocess
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt


class Model(tf.keras.Model):
    def __init__(self, classification, is_lstm):
        super(Model, self).__init__()
        #Define batch size
        self.batch_size = 84
        self.linear_size_one = 256 
        self.linear_size_two = 84 
        
        #Define embedding matrix
        self.embedding = tf.keras.layers.Embedding(450000, 128)
        
        #Define layers
        if is_lstm:
            self.model_layer = tf.keras.layers.LSTM(self.linear_size_one, return_sequences=True)
        else:
        #For GRU Model
            self.model_layer = tf.keras.layers.GRU(self.linear_size_one, return_sequences=True)

        self.l1 = tf.keras.layers.Dense(self.linear_size_two, activation='relu')
        #Pass in the number of output classes
        self.l2 = tf.keras.layers.Dense(classification, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam()
        #Loss list for plotting the losses
        self.loss_list = []

    def call(self, reviews):
        #The shape of the self.embedding output will be [sentence_length, batch_size, embedding_dim]
        l1_out = self.embedding(reviews)

        #Pass inputs through LSTM or GRU
        l2_out = self.model_layer(l1_out)

        l2_out = tf.reshape(l2_out, (self.batch_size, -1))
        
        #Pass through dense layers
        l3_out = self.l1(l2_out)
       
       
        final_out = self.l2(l3_out) 
        
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
            #Returns the indices of the maximum value thus if correctly predicted increments counter
            if tf.argmax(predictions[i]).numpy() == labels[i]:
                correct_count += 1
            count += 1
        
        #Return the correct predictions over total to give accuracy for that batch
        return correct_count/count

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

    #Change classification and model
    model = Model(classification=5, is_lstm=False)

    #Get the train and test inputs and labels from preprocess
    #Pick classification
    train_inputs, test_inputs, train_labels, test_labels = preprocess(classification=5)

    train(model, train_inputs, train_labels)
    accuracy = test(model, test_inputs, test_labels)
    print("accuracy", accuracy)
    visualize_loss(model.loss_list)
    
     
if __name__ == "__main__":
    main()
