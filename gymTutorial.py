import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Dense
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_dimensions, action_size):
        self.state_dimensions = state_dimensions
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model


        model = Sequential()
        # activation(dot(input, kernel) + bias) Three fully connected layers used here
        # this is a pretty sweet deal for this to be reduced to just this
        model.add(Conv2D(128, (3, 3), input_shape=self.state_dimensions, activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))


        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # this keeps track of the state
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #this is the random choice of acts
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state)
        print("It reaches this point")
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        #This is where reward happens
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print("state:", state)
            print("action:", action)
            print("reward:", reward)
            print("next_state:", next_state)
            print("done:", done)
            # target reward
            target = reward
            if not done:
                # This predicts the reward
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    state_dimensions = env.observation_space.shape
    print(env.action_space)
    action_size = env.action_space.n
    state = env.reset()
    print("state:", state)
    agent = DQNAgent(state_dimensions, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        print("state:", state)
        #state = np.reshape(state, [4, ])
        #state = [1, 2, 3, 4]
        while not done:
            #print("going")
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            #next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, e: {:.2}"
                      .format(e, EPISODES, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(10)