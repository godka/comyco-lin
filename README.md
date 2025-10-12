# Comyco for linear-based QoE

This is an official implementation of Comyco for $QoE_\texttt{lin}$. Compared with the original Comyco, we made several changes here.

- *Instant solver*: we upgrade the "MPC-based solver" to "Beam-search solver", which extends the future look horizon to **15** (default: 8). In other words, by leveraging beam search technologies, Comyco can obtain more precise expert strategies with lower execution costs.

- *Discounted bandwidth*: Considering that Comyco sometimes tends to pick higher bitrates due to the almost omniscient Instant Solver, we integrate a discount factor of 0.9 for computing bandwidth. This has proven to be effective in helping Comyco avoid rebuffering events.

- *Neural network*: For the sack of fairness, Comyco's neural network is aligned with the Pensieve's.
- *Framework update*: The learning pipeline has been migrated from TensorFlow/TFLearn to PyTorch.

- *Loss function*: We adjust the weight of entropy to 0.1 for reducing exploration. Note that this differs slightly from the description in the original paper.

# Build by yourself

We use pybind11 to implement a beam-search solver. Build the native extension with setuptools:

```
pip install pybind11 setuptools
python core/setup.py build_ext --inplace
```

This will produce a `libcore` module in the repository root that is compatible with the PyTorch
training pipeline. The pre-build version was built using Python 3.7.9 and runs on Ubuntu 20.04.

# Train Comyco

Training now relies on PyTorch. After installing the dependencies (PyTorch, NumPy and tqdm), launch
the training loop with:

```
python train.py
```

TensorBoard summaries are written to `./comyco`, so you can monitor learning with:

```
tensorboard --logdir comyco
```

We have saved one of the training logs in the `./logs/` directory for reference.

# Test Comyco

Run the evaluation script against a PyTorch checkpoint:

```
python test.py path_to_your_model
```

We provide a pre-trained model for convenience:

```
python test.py pretrained/model
```

# Evaluation

plot the figure using

```
pip install matplotlib
python plot.py
```

Results are reported in 'imgs/cdf.png'. As shown, Comyco improves the average QoE of 6.5\% compared to Pensieve.


<p align="center">
    <img src="imgs/norway.png" alt="HSDPA" width="40%">
</p>
<p align="center">
    <em>HSDPA</em>
</p>
<p align="center">
    <img src="imgs/fcc.png" alt="FCC" width="40%">
</p>

<p align="center">
    <em>FCC</em>
</p>
<p align="center">
    <img src="imgs/oboe.png"  alt="Oboe" width="40%">
</p>

<p align="center">
    <em>Oboe</em>
</p>
