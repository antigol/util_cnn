# util_cnn

Utilities to train and compare classifiers neural networks coded with PyTorch

    python3 setup.py install --user

## Example

Create your `model.py` like that:
```python
import torch
import numpy as np
from util_cnn.model import Model

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  # pylint: disable=W0221
        # Your amazing CNN
        return x


class MyModel(Model): # This class must be called MyModel
    def __init__(self):
        super().__init__()
        self.cnn = CNN()

    def get_cnn(self):
        return self.cnn

    def get_batch_size(self):
        return 128

    def get_learning_rate(self, epoch):
        return 1e-2

    def load_files(self, files):
        return torch.FloatTensor(np.array([np.load(f) for f in files]))
```

Create `train.csv` like that
```csv
id,class
1000,cat
1001,cat
1002,dog
1999,cat
```
In the directory `data/` you have the files `data/1001.npy`, `data/1002.npy` and so on

Same idea for the test set.

Then it can be trained with `train.py`

    python3 util_cnn/train.py --number_of_epochs 1000 --train_data_path "data/*.npy" --train_csv_path train.csv --eval_data_path "data/*.npy" --eval_csv_path test.csv --log_dir my_first_training --model_path model.py
    
It will train with the train data and evaluate with the test data and put all logs and parameters into a directory called `my_first_training`.

To see other options type:

    python3 util_cnn/train.py --help
