# PPO Pytorch C++

This is an implementation of the [proximal policy optimization algorithm](https://arxiv.org/abs/1707.06347) for the C++ API of Pytorch. It uses a simple `TestEnv` in `main.cpp` to test the algorithm.

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

## Results
The results are save to `build/data.csv` and can be visualized by running `python plot.py`.
