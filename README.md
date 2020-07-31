# HIL_mountainCar
Hierarchical imitation learning for the MountainCar openAI-Gym environment

# Main
- expert_recorder.py
- BC_main.py  (for Behavioral Cloning)
- HIL_main (for Hierarchical Imitation Learning)
- main (for regularizers validation)

## Expert_recorder
We need first to play the game and record the data. To do that, open the terminal and type

```bash
python expert_recorder.py Expert/Data

```

## BC_main
```python
import HierarchicalImitationLearning as hil
import BehavioralCloning as bc
import tensorflow as tf
import gym
import numpy as np
import Simulation as sim
import matplotlib.pyplot as plt
```

### Pipeline
- Sample data from Expert
- Train Neural Networks for BC
  - NN1 uses Cross Entropy loss function
  - NN2 Mean Squared Error
  - NN3 Hinge Loss
- Evaluate Performance of NNs with different number of Training samples
- Plot distribution learnt by NN

## HIL_main

### Dependencies
```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib as mpl
import Simulation as sim
import HierarchicalImitationLearning as hil
import gym
import BehavioralCloning as bc
import concurrent.futures
```

### Pipeline
- Sample data from Expert
- Initialize Hierarchical inverse learning hyperparameters
- Fix gains for regularization (lambdas, eta)
- Train the Triple of NNs using Baum-Welch with failure algorithm
- Save Trained model
- Load Model
- Evaluate Performance

## main

### Dependencies
```python
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
import HierarchicalImitationLearning as hil
import numpy as np
import matplotlib.pyplot as plt
import Simulation as sim
import BehavioralCloning as bc
import concurrent.futures
import gym
from joblib import Parallel, delayed
import multiprocessing
```

### Pipeline
- Generate Expert's policy
- Sample data from Expert
- Initialize Hierarchical inverse learning hyperparameters
- Understanding Regularization 1
- Understanding Regularization 2
- Reguralizers validation (lambdas, eta)
- Determine best Policy
- Save Trained model
- Evaluation 
- Evaluation of the training on different number of samples 


### Important
This code runs the validation of the regularizers generating multithreads using "joblib.Parallel".
Run might take several minutes and cannot be killed manually once started; therefore, select the hyperparameters appropriately.
