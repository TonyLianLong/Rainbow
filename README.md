# An Rainbow implementaion with PARL
This is an rainbow implementation which uses [PARL](https://github.com/PaddlePaddle/PARL) and its backend for PyTorch.
Currently you could run the following command to train and test:

```
python main.py
```

After 20000 steps, you will likely have:
```
[07-01 18:19:54 MainThread @DQN_agent.py:190] Frame: 20000, Score: 169.2, loss: 4.88

[07-01 18:20:02 MainThread @DQN_agent.py:179] score: 200.0
```

# Demo
## Start
Note that the angle becomes larger than 15 deg quickly.
![start](https://user-images.githubusercontent.com/1451234/86233373-e5697080-bbc7-11ea-934a-914ed054921d.gif)

## End
Note that even after 200 units of time the cart is still on the screen.
![end](https://user-images.githubusercontent.com/1451234/86233363-e26e8000-bbc7-11ea-8307-733e0096b4f0.gif)

# Note
This project is heavily inspired by [Rainbow is what you need](https://github.com/Curt-Park/rainbow-is-all-you-need). The PARL document and source code are also referenced in the development of this project.

In addition, PARL selects backend based on whether paddlepaddle is installed, and paddlepaddle backend has a higher priority compared to pytorch backend.

To use pytorch backend, you need to have pytorch installed and paddlepaddle not installed.
