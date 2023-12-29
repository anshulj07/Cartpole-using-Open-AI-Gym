import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.optimizers import legacy
from keras import layers



def OurModel(input_shape, action_space):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(action_space, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss="mse", optimizer=legacy.RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole', render_mode='human')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95 
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 1000

        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
            
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = state[0] if isinstance(state, tuple) else state 
            print("State shape:", state.shape)
            print("State type:", type(state))
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    if i == 500:
                        print("Saving trained model as CartPole")
                        self.save("CartPole")
                        return
                self.replay()

        def load(self, name):
            self.model = load_model(name)

        def save(self, name):
            self.model.save(name)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def test(self):
        for e in range(self.EPISODES):
            self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
            state = self.env.reset()
            state = state[0] if isinstance(state, tuple) else state
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break

if __name__ == "__main__":
    agent = DQNAgent()
    #agent.run()
    agent.test()
