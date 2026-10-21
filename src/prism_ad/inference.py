"""Deterministic inference without changing fitted parameters or training data."""
import numpy as np
import torch

def seeded_predict(model,x,fold):
    # The package's randomized GPU SVD uses the global Torch RNG at predict time.
    # Fix that call's random stream and restore the caller's CPU/CUDA RNG states.
    cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(0)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(42+fold);torch.cuda.manual_seed_all(42+fold)
        p=model.predict_proba(x)[:,list(model.classes_).index(1)].astype(float)
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(0))
    return p
