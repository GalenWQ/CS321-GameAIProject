import os
from warnings import filterwarnings

# Suppress warnings that are impossible to fix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filterwarnings("ignore", ".*compiletime version.*tensorflow.*", RuntimeWarning)
from sys import argv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, TimeDistributed
from keras.optimizers import Adam
from keras.utils import plot_model, print_summary
from environment import env


class FiniteBuffer:
    """Finite memory buffer for DQNAgent's memory. Would have used deque, but it has slow random indexing."""
    def __init__(self, size, initial_data=None):
        """Construct a finite size buffer"""
        if not isinstance(size, int) or size < 0:
            raise TypeError("Size must be non-negative integer")
        self.size = size
        self.index = 0
        self.is_full = False
        self.data = [None for x in range(size)]
        if initial_data:
            for item in initial_data:
                self.append(item)

    def append(self, item):
        if self.index == self.size:
            self.is_full = True
            self.index = 0
        self.data[self.index] = item
        self.index += 1

    def get_sample(self, k):
        """Return k randomly selected items from the buffer"""
        end = None if self.is_full else self.index
        return random.sample(self.data[:end], k)

    def __len__(self):
        if self.is_full:
            return self.size
        return self.index


class DQNAgent:
    """Class for a DQN agent."""

    def __init__(self, state_shape, state_dtype, frames_per_observation, action_size):
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self.action_size = action_size
        self.m = frames_per_observation
        self.obs_shape = (self.m,) + state_shape
        self.mem_dtype = np.dtype(
            [("observation", state_dtype, self.obs_shape), ("action", np.uint8), ("reward", np.uint16),
             ("next_observation", state_dtype, self.obs_shape), ("done", np.bool_)])
        self.buffer = FiniteBuffer(size=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Builds a Keras Sequential model for neural net Deep-Q learning."""


        # Apply conv_1 to each frame, separately, with TimeDistributed wrapper
        conv_1 = Conv2D(20, 8, strides=4, input_shape=self.state_shape, activation='relu')
        conv_2 = Conv2D(48, 4, strides=2, activation='relu')
        conv_3 = Conv2D(48, 3, strides=1, activation='relu')
        dense = Dense(128, activation='relu')
        final = Dense(self.action_size, activation='linear')
        input_ = TimeDistributed(conv_1, input_shape=self.obs_shape)
        model = Sequential((input_, TimeDistributed(conv_2), TimeDistributed(conv_3), TimeDistributed(Flatten()),
                            TimeDistributed(dense), Flatten(), final))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, observation, action, reward, next_observation, done):
        """Add the given decision description to memory."""
        memory = np.array((observation, action, reward, next_observation, done), self.mem_dtype)
        self.buffer.append(memory)

    def act(self, observation):
        """Randomly either return a random action or prediction from model, based on exploration rate"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        # Note: "state[None]" is the same as "np.array([state])" or "np.expand_dims(state, 0)", but faster
        predicted_rewards = self.model.predict(observation[None])[0]
        return np.argmax(predicted_rewards)

    def replay(self, batch_size):
        """This is where reward happens"""
        memories = np.array(self.buffer.get_sample(batch_size))
        memories = memories.transpose()
        observations = memories["observation"]
        next_observations = memories["next_observation"]
        predicted_rewards = self.gamma * np.amax(self.model.predict(next_observations), axis=1)
        predicted_rewards *= memories["done"].astype(np.uint8)
        target_rewards = memories["reward"] + predicted_rewards
        target_outputs = self.model.predict(observations)
        for i in range(batch_size):
            target_outputs[i][memories["action"][i]] = target_rewards[i]
        # v = 0 if random.random() <= .9 else 1
        v = 0
        self.model.fit(observations, target_outputs, batch_size=batch_size, verbose=v)
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
        plot_model(self.model, "model.png")


def preprocess_image(frame):
    frame = np.average(frame, axis=-1, weights=[.2125, .7154, .0721]).astype(np.uint8)[..., None]
    return frame[::2, ::2]


def main():
    """Uses command line arguments for save/load filename for weights."""

    if len(argv) <= 1:
        raise ValueError("No save filename given.")
    if len(argv) >= 3:
        episodes = int(argv[2])
    else:
        episodes = 10000
    save_file_name = argv[1]
    frames_per_observation = 4
    # agent = DQNAgent(env.observation_space.shape, env.observation_space.dtype, frames_per_observation,
    #                  env.action_space.n)
    space = preprocess_image(env.reset())
    agent = DQNAgent(space.shape, space.dtype, frames_per_observation, env.action_space.n)
    if len(argv) >= 4:
        agent.load(argv[3])
    print_summary(agent.model)
    plot_model(agent.model, to_file="model.png")

    for e in range(episodes):
        score = 0
        last_observation = np.array([preprocess_image(env.reset())] * 4)
        last_obs_reward = 0
        done = False
        while not done:
            action = agent.act(last_observation)
            obs_reward = 0
            observation = []
            for i in range(frames_per_observation):
                # Rendering takes very little time compared to learning.
                env.render()
                frame, reward, done, _ = env.step(action)
                processed = preprocess_image(frame)
                observation.append(processed)
                obs_reward += reward
                # If environment is Breakout, then score is sum of rewards.
                score += int(reward)
            observation = np.array(observation)
            # display_obs(observation)
            agent.remember(last_observation, action, last_obs_reward, observation, done)
            last_observation = observation
            last_obs_reward = obs_reward
        print("episode: {}/{}, e: {:.2}, score: {}".format(e, episodes, agent.epsilon, score))
        batch_size = min(128, len(agent.buffer))
        agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(save_file_name)
            print("buffer: {}/ {}. size: {} mb".format(len(agent.buffer), agent.buffer.size,
                                                       agent.buffer.data[0].nbytes * len(agent.buffer) / (10 ** 6)))

if __name__ == "__main__":
    main()
