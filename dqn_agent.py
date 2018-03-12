from sys import argv
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam


class DQNAgent:
    """Class for a DQN agent."""

    def __init__(self, state_dimensions, action_size):
        self.state_dimensions = state_dimensions
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # TODO: consider changing max length
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Keras Sequential model for neural net Deep-Q learning."""
        model = Sequential()
        # model.add(Conv2D(210, 4, strides=3, input_shape=self.state_dimensions, activation='relu'))
        # model.add(MaxPooling2D(pool_size=(3, 3)))
        # model.add(Conv2D(64, 3, strides=3, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))

        # Intended these layers to be a smaller version of what the DeepMind team did
        # TODO: play around with this.
        model.add(Conv2D(20, 6, strides=3, input_shape=self.state_dimensions, activation='relu'))
        model.add(Conv2D(48, 4, strides=2, activation='relu'))
        model.add(Conv2D(48, 3, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Add the given decision description to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Randomly either return a random action or prediction from model, based on exploration rate"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        # Note: "state[None]" is the same as "np.array([state])" or "np.expand_dims(state, 0)", but faster
        act_values = self.model.predict(state[None])[0]
        return np.argmax(act_values)

    def replay(self, batch_size):
        """This is where reward happens"""
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            # print("state:", state)
            # print("action:", action)
            # print("reward:", reward)
            # print("next_state:", next_state)
            # print("done:", done)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[None]))
            target_f = self.model.predict(state[None])
            target_f[0][action] = target
            self.model.fit(state[None], target_f, batch_size=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def layer_description(self):
        """Prints input/output format of each layer in the agent's model. Useful for testing."""
        for layer in self.model.layers:
            print("input: {},\noutput: {}".format(layer.input, layer.output))


def main():
    """Uses command line arguments for save/load filename for weights. Trains agent on Breakout-v0"""
    if len(argv) <= 1:
        raise ValueError("No save filename given.")
    save_file_name = argv[1]

    env = gym.make('Breakout-v0')
    state_dimensions = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_dimensions, action_size)
    if len(argv) >= 3:
        episodes = int(argv[2])
    else:
        episodes = 1000
    if len(argv) >= 4:
        agent.load(argv[3])

    # TODO: think about batch size
    batch_size = 48

    for e in range(episodes):
        score = 0
        state = env.reset()
        done = False
        while not done:
            # Rendering takes very little time compared to learning.
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # If environment is Breakout, then score is sum of rewards.
            if env.spec == gym.spec("Breakout-v0"):
                score += int(reward)

        print("episode: {}/{}, e: {:.2}, score: {}".format(e, episodes, agent.epsilon, score))
        agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(save_file_name)


if __name__ == "__main__":
    main()
