import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import permutation
from templates import utils

M = 10
eps = 1e-10

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    # permutation
    print("Validating permutation")
    for m in range(1, M+1):
        X, Y_train = permutation.get_dataset(m)
        X_train = jnp.eye(len(X))
        Y_train = jnp.array(Y_train, dtype=jnp.float64)
        n0 = len(X)
        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),   # First dense layer with 128 units
            stax.Relu(),       # ReLU activation
            stax.Dense(n0)      # Output dense layer with 1 unit
        )
        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)

        numbers = list(range(2**m))

        for i in tqdm(range(len(numbers))):    
            
            p = numbers[i]
            X_test = permutation.get_sample(m)[1]
            X_test['p'] = np.array([int(x) for x in np.binary_repr(p, width=m)])[::-1].tolist()

            X_test = utils.encode_data(X_test, X)

            X_test = jnp.array(X_test, dtype=jnp.float64).reshape(1, -1)
            y_pred = predict_fn(x_test=X_test, get='ntk', compute_cov=True)
            
            y_pred_round = np.where(y_pred.mean[0] > eps, 1.0, 0.0).tolist()
            X_test = permutation.unflatten_sample(y_pred_round, m)

            out_bin = np.array([int(x) for x in np.binary_repr(p, width=m)])[::-1].tolist()
            assert out_bin == X_test['p'] 

        print(f"Permutation - bit length: {m} - all correct")
    print("Permutation complete")