# TorchKart

data
----
We prepare 10 races on Luigi Raceway for training.
```
data/0_x.npy
data/0_y.npy
...
data/9_x.npy
data/9_y.npy
```
If you want to record your own data, run ```python3 record.py [id]``` will generate ```[id]_x.npy``` and ```[id]_y.npy```.

Training
----
```
python3 main.py --train
```
The program default load prepared 10 races ad training data, and save ```pre-train.pt``` for testing.

Testing
----
```
python3 main.py --test
```
The Program will load ```pre-train.pt``` and plays on Luigi Raceway.
