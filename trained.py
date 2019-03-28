import gym
import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
from numpy.random import choice

tf.reset_default_graph()
LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()




class neural_network:

    def __init__(self):
        self.graph = tf.Graph()
        tf.reset_default_graph()
        with self.graph.as_default():
               tflearn.config.init_training_mode()
               self.input = input_data(shape=[None, 4,1])
               self.l1= fully_connected(self.input, 100, activation='relu')
               self.l2= fully_connected(self.l1, 50, activation='relu')
               self.l3 = fully_connected(self.l2, 2, activation='softmax')
               self.r = regression(self.l3, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')

               self.model = tflearn.DNN(self.r)

    def predict(self,data):
        with self.graph.as_default():
            prediction = self.model.predict(data)
        return prediction

    def cross(self,other):
        child = neural_network()
        # warstwa pierwsza

        parent1_w = self.model.get_weights(self.l1.W)
        parent2_w = other.model.get_weights(other.l1.W)

        child_w = parent1_w


        for i in range(len(parent1_w)):
            for j in range(len(parent1_w[i])):
                child_w[i][j] = random.choice([parent1_w[i][j],parent2_w[i][j]])

        child.model.set_weights(child.l1.W,child_w)

        # wartstwa druga

        parent1_w = self.model.get_weights(self.l2.W)
        parent2_w = other.model.get_weights(other.l2.W)

        child_w = parent1_w



        for i in range(len(parent1_w)):
            for j in range(len(parent1_w[i])):
                child_w[i][j] = random.choice([parent1_w[i][j],parent2_w[i][j]])


        child.model.set_weights(child.l2.W,child_w)

        # trzecia warstwa

        parent1_w = self.model.get_weights(self.l3.W)
        parent2_w = other.model.get_weights(other.l3.W)

        child_w = parent1_w



        for i in range(len(parent1_w)):
            for j in range(len(parent1_w[i])):
                child_w[i][j] = random.choice([parent1_w[i][j],parent2_w[i][j]])


        child.model.set_weights(child.l3.W,child_w)
        return child
    def load(self,filename):
        with self.graph.as_default():
            self.model.load(filename,weights_only=True)

network = neural_network()
network.load("./epoch9.tfl")
model = network




for each_game in range(10):
    env.reset()
    observation, reward, done, info = env.step(1)
    for _ in range(200):
        env.render()
        action = np.argmax(model.predict(observation.reshape(-1,len(observation),1))[0])

        observation, reward, done, info = env.step(action)

        if done: break
