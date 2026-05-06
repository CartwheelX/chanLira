## QuRiFT Experiment Drivers

The central QuRiFT entry point is:

```bash
python experiments/qurift_main.py
```

The sweep drivers in this directory call `qurift_main.py` for controlled
structural privacy experiments across synthetic datasets and MNIST:

```text
full_sweep_qnn_moons.py
full_sweep_qnn_circles.py
full_sweep_qnn_blobs.py
run_mnist_sweep_qnn.py
run_mnist_sweep_hqnn.py
run_mnist_sweep_qcnn.py
```

TorchQuantum provides the upstream quantum primitives used by the models.
