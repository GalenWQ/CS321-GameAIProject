"""Modified from tutorial code. Contains class for a DQN agent, and function to test that class"""
from sys import argv
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D
from keras.optimizers import Adam

EPISODES = 10000


class DQNAgent:
    """Class for a DQN agent"""
    def __init__(self, state_dimensions, action_size):
        self.state_dimensions = state_dimensions
        self.action_size = action_size
        self.memory = deque()  # note: removed maxlen
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Keras Sequential model for neural net Deep-Q learning."""
        model = Sequential()
        # activation(dot(input, kernel) + bias) Three fully connected layers used here
        # this is a pretty sweet deal for this to be reduced to just this


        # TODO: first experiment with window size of first layer 128, 210, 420 (first just get it running)
        # second: with the best result, change kernel size to 4,6,8
        # third: figure out what its saving
        # fourth: figure out how to stop from rendering
        # fifth: comment up 
        
        model.add(Conv2D(210, 4, strides=3, input_shape=self.state_dimensions, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(64, 3, strides=3, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Keeps track of the state"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Randomly either return a random action or prediction from model, based on exploration rate"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0][0][0])  # returns action

    def replay(self, batch_size):
        """This is where reward happens"""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            #print("state:", state)
            #print("action:", action)
            #print("reward:", reward)
            #print("next_state:", next_state)
            #print("done:", done)
            # target reward
            target = reward

            if not done:
                # This predicts the reward
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main():
    if len(argv) <= 1:
        raise ValueError("No save filename given.")
    save_file_name = argv[1]
    
    env = gym.make('Breakout-v0')
    state_dimensions = env.observation_space.shape
    action_size = env.action_space.n
    state = env.reset()
    #print("state:", state)
    agent = DQNAgent(state_dimensions, action_size)
    if len(argv) >= 3:
        agent.load(argv[2])
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        print("state.shape:", state.shape)
        #state = np.reshape(state, [4, ])
        state = np.expand_dims(state, axis=0)
        print("expanded state.shape:", state.shape)
        input()
        #state = [1, 2, 3, 4]
        while not done:
            #print("going")
            env.render()
            action = agent.act(state)
            #print("action:", action)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            #print("new reward:", reward)
            #next_state = np.reshape(next_state, [1, 4])
            next_state = np.expand_dims(next_state, axis=0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                #env.reset()
                print("episode: {}/{}, e: {:.2}"
                      .format(e, EPISODES, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(10)

        done = False

        if e % 10 == 0:
            agent.save(save_file_name)
            # agent.save("save.h5")            
    
if __name__ == "__main__":
    main()
   
