# Execution environments

The original analysis used Python 3.11.15, PyTorch 2.6.0 with CUDA 12.4,
TabPFN 8.2.0 and the versions in model-requirements.txt. Full comparator
training uses an NVIDIA GPU and separately obtained licensed TabPFN-3
weights; no model weights or credentials are distributed here. CPU model
tests use the same equations with synthetic inputs.

Figure reconstruction was tested with Python 3.8, NumPy 1.22.3,
pandas 1.3.5 and Matplotlib 3.5.2. Nimbus Sans Regular/Bold from URW Base35
are needed to reproduce the original typography. Install the figure
requirements in a separate environment; do not force these older packages
into the model environment.

These files record tested package versions, not a complete cross-platform
dependency lock. Hardware, BLAS, font files and GPU libraries can affect
byte-level reproducibility. See the execution records accompanying the
prepared source-data package for the tested output comparisons.
