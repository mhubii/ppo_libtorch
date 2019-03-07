# PPO Pytorch C++

This is an implementation of the [proximal policy optimization algorithm](https://arxiv.org/abs/1707.06347) for the C++ API of Pytorch. It uses a simple `TestEnv` in `main.cpp` to test the algorithm. Below is a small visualization of the environment, the algorithm is tested in.

<br>
<figure>
  <p align="center"><img src="img/epoch_1.gif" width="40%" height="40%" hspace="40"><img src="img/epoch_5.gif" width="40%" height="40%" hspace="40"><img src="img/epoch_20.gif" width="40%" height="40%" hspace="40"></p>
  <figcaption>Fig. 1: From top left to bottom right, the agent as it takes actions in the environment to reach the goal. </figcaption>
</figure>
<br><br>

## Build
Do
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolut/path/to/libtorch ..
make
```
Please check out [Pytorch](https://pytorch.org/cppdocs/installing.html#minimal-example) to find out how to build against the C++ API.

## Run
Run the executable and specify the x, and y-position of the goal
```
./testPPO goal_x goal_y
```
for example
```
./testPPO 2 2
```

## Visualization
The results are saved to `data/data.csv` and can be visualized by running `python plot.py`.
