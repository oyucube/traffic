import numpy as np
import socket
xp = np
env_test = "cpu"
if socket.gethostname() == "chainer":
    import cupy as cp
    xp = cp
    env_test = "gpu"
