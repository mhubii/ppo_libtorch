# PPO Pytorch C++

This is an implementation of the [proximal policy optimization algorithm](https://arxiv.org/abs/1707.06347) for the C++ API of Pytorch. It uses a simple `TestEnvironment` to test the algorithm. Below is a small visualization of the environment, the algorithm is tested in.
<br>
<figure>
  <p align="center"><img src="img/test_mode.gif" width="50%" height="50%" hspace="0"></p>
  <figcaption>Fig. 1: The agent in testing mode. </figcaption>
</figure>
<br><br>

## Build
You first need to install PyTorch. For a clean installation from Anaconda, checkout this short [tutorial](https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5), or this [tutorial](https://pytorch.org/cppdocs/installing.html), to only install the binaries.

Do
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolut/path/to/libtorch ..
make
```

## Run
Run the executable with
```
cd build
./train_ppo
```
It should produce something like shown below.
<br>
<figure>
  <p align="center"><img src="img/epoch_1.gif" width="50%" height="50%" hspace="0"><img src="img/epoch_10.gif" width="50%" height="50%" hspace="0"></p>
  <figcaption>Fig. 2: From left to right, the agent for successive epochs in training mode as it takes actions in the environment to reach the goal. </figcaption>
</figure>
<br><br>

The algorithm can also be used in test mode, once trained. Therefore, run
```
cd build
./test_ppo
```
## Visualization
The results are saved to `data/data.csv` and can be visualized by running `python plot.py`.
