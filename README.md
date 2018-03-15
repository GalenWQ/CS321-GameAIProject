# CS321-GameAIProject

Gym documentation:

https://github.com/openai/gym


Basic gym tutorial (with setup information):

https://medium.freecodecamp.org/how-to-build-an-ai-game-bot-using-openai-gym-and-universe-f2eb9bfbb40a

Sample code:

https://github.com/harinij/100DaysOfCode/tree/master/Day%2022

To run this code, install gym[atari], numpy and keras, then just run "python3 dqn_agent.py"
Installing the atari part of gym on windows is possible but very annoying. See https://github.com/j8lp/atari-py/blob/master/README.md
To save the model's weights after training, give the name of the file to save to as the first command line argument.
To specify a number of episodes other than 10000, give it as the second argument. To load weights from a previously saved
file for the same model topology, give the name of the file to load from as a third argument.