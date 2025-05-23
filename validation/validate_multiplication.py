import numpy as np
import jax
import jax.numpy as jnp
import neural_tangents as nt

from neural_tangents import stax
from itertools import product
from tqdm import tqdm

from templates import multiplication
from templates import utils

M = 10
eps = 1e-10

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":

    print("Validating multiplication")
    for m in range(1, M+1):

        X, Y_train = multiplication.get_dataset(m)
        pad = m
        X_train = jnp.eye(len(X) + pad)
        Y_pad = np.zeros((pad, Y_train.shape[1]), dtype=np.float64)
        Y_train = jnp.array(np.vstack((Y_train, Y_pad)), dtype=jnp.float64)

        n0 = X_train.shape[0]

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024),   # First dense layer with 128 units
            stax.Relu(),       # ReLU activation
            stax.Dense(n0)      # Output dense layer with 1 unit
        )
        predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, X_train, Y_train)

        pairs = product(list(range(2**m)), list(range(2**m)))

        for i in tqdm(range(2**(2*m))):
            multiplier, multiplicand = next(pairs)
            out = multiplier*multiplicand

            X_test = multiplication.get_sample(m)[1]
            X_test['multiplier'] = np.array([int(x) for x in np.binary_repr(multiplier, width=m)])[::-1].tolist()
            X_test['multiplicand'] = np.array([int(x) for x in np.binary_repr(multiplicand, width=2*m)])[::-1].tolist()
            X_test['to_check_lsb'][0] = 1

            for i in range(4*(m**2)+ 3*m):
                X_test = utils.encode_data(X_test, X)
                X_test = np.concatenate([X_test, np.zeros(pad)])
                X_test = jnp.array(X_test, dtype=jnp.float64).reshape(1, -1)
                y_pred = predict_fn(x_test=X_test, get='ntk', compute_cov=True)

                y_pred = np.where(y_pred.mean[0] > eps, 1, 0).tolist()
                X_test = multiplication.unflatten_sample(y_pred, m)
                #print(X_test)

            out_bin = np.array([int(x) for x in np.binary_repr(out, width=2*m)])[::-1].tolist()

            assert out_bin == X_test['sum_p']

        print(f"Multiplication - Bit length: {m} - all correct")
    print("Multiplication complete")