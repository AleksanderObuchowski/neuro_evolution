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






def random_games():

    for episode in range(15):
        env.reset()

        for t in range(200):

            env.render()


            action = env.action_space.sample()


            observation, reward, done, info = env.step(action)
            if done:
                break


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


def evolution(population):
    fitness = []
    for i in range(len(population)):
        env.reset()
        observation , reward, done, info = env.step(1)
        fitness.append(0)
        for j in range(200):
            if(done):
                break
            action = np.argmax(population[i].predict(observation.reshape(1,len(observation),1)))
            observation, reward, done, info = env.step(action)
            fitness[i]+= reward
    children = []
    fitness = np.array(fitness)
    fitness= fitness/fitness.sum(axis=0,keepdims=1)
    for i in range(100):
        parent1 = choice(population, 1, p=fitness)
        parent2 = choice(population, 1, p=fitness)
        children.append(parent1[0].cross(parent2[0]))
    return children,fitness


def train_model():

    population = []
    n = neural_network()
    print("creating polulation")
    for _ in range(100):
        n = neural_network()
        population.append(n)
    print("population created")

    for e in range(10):
        print("Epoch: ",e)
        population,fitness = evolution(population)

        keydict = dict(zip(population, fitness))
        population.sort(key=keydict.get)

        population[-1].model.save("epoch"+ str(e) +".tfl")

    return population[-1]



model = train_model()




for each_game in range(10):
    env.reset()
    observation, reward, done, info = env.step(1)
    for _ in range(200):
        env.render()
        action = np.argmax(model.predict(observation.reshape(-1,len(observation),1))[0])
        print(action)
        observation, reward, done, info = env.step(action)

        if done: break
