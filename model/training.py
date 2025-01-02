import time
import numpy as np
from numba import njit

@njit(cache=True)
def rowwise_softmax(v):
    for i in range(v.shape[0]):
        mx = v[i, 0]
        for j in range(1, v.shape[1]):
            if v[i, j] > mx:
                mx = v[i, j]
        s = 0.0
        for j in range(v.shape[1]):
            v[i, j] = np.exp(v[i, j] - mx)
            s += v[i, j]
        for j in range(v.shape[1]):
            v[i, j] /= s

@njit(cache=True)
def rowwise_cross_entropy(prob, targets):
    eps = 1e-9
    loss_sum = 0.0
    for i in range(targets.shape[0]):
        p = prob[i, targets[i]]
        if p < eps:
            p = eps
        loss_sum -= np.log(p)
    return loss_sum / targets.shape[0]

@njit(cache=True)
def _train_loop(emb, proj, batches, epochs, lr, debug):
    epoch_losses = np.zeros(epochs, dtype=np.float64)
    for ep in range(epochs):
        s = 0.0
        c = 0
        for x, y in batches:
            v = emb[x] @ proj
            rowwise_softmax(v)
            loss = rowwise_cross_entropy(v, y)
            s += loss
            c += 1
            for i in range(y.shape[0]):
                v[i, y[i]] -= 1.0
            inv_bsz = 1.0 / y.shape[0]
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    v[i, j] *= inv_bsz
            d = v @ proj.T
            for i in range(y.shape[0]):
                emb[x[i]] -= lr * d[i]

            if debug:
                print("Batch Loss =", loss)

        epoch_losses[ep] = s / c
        if debug:
            print(f"Epoch {ep+1}, Avg Loss = {epoch_losses[ep]}")
    return epoch_losses

@njit(cache=True)
def _train_single_epoch(emb, proj, batches, lr, debug):
    s = 0.0
    c = 0
    for x, y in batches:
        v = emb[x] @ proj
        rowwise_softmax(v)
        loss = rowwise_cross_entropy(v, y)
        s += loss
        c += 1
        for i in range(y.shape[0]):
            v[i, y[i]] -= 1.0
        inv_bsz = 1.0 / y.shape[0]
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                v[i, j] *= inv_bsz
        d = v @ proj.T
        for i in range(y.shape[0]):
            emb[x[i]] -= lr * d[i]
        if debug:
            print("Batch Loss =", loss)
    avg_loss = s / c if c > 0 else 0.0
    return avg_loss

def train_model(model, data, vocab_size, epochs, lr, debug=False, estimate_fraction=0.3):
    emb = model.embeddings
    proj = model.projection
    batches = [(np.array(inp, dtype=np.int32),
                np.array(tgt, dtype=np.int32)) for inp, tgt in data]

    if len(batches) > 0 and epochs > 0:
        sample_size = max(1, int(len(batches) * estimate_fraction))
        sample_batches = batches[:sample_size]

        emb_trial = emb.copy()
        proj_trial = proj.copy()

        start_est = time.perf_counter()
        _train_single_epoch(emb_trial, proj_trial, sample_batches, lr, debug=False)
        end_est = time.perf_counter()

        trial_time = end_est - start_est
        fraction = sample_size / len(batches)
        one_epoch_time_est = trial_time / fraction
        total_est = one_epoch_time_est * epochs

        print(f"Estimated total training time: {total_est:.2f} seconds "
              f"({total_est/60:.2f} minutes) for {epochs} epochs.\n")
    else:
        print("Insufficient data or epochs=0; skipping time estimate.\n")

    start_time = time.perf_counter()
    losses = _train_loop(emb, proj, batches, epochs, lr, debug)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_epoch_time = total_time / epochs if epochs > 0 else 0

    print(f"\nTraining completed in {total_time:.2f} seconds for {epochs} epochs.")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds per epoch.")

    if debug:
        for i, loss_val in enumerate(losses, start=1):
            print("Epoch", i, "Loss =", loss_val)
