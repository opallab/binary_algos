import math
import argparse
import os
import torch
import dataset
from model import NTKMLP
from torch.utils.data import DataLoader


def train_model(train_loader: DataLoader, epochs: int, k: int, hidden_dim: int, sigma_w: float = None, sigma_b: float = None, device: str = 'cpu', lr: float = 1e-3):
    """
    Train a model on the given dataset.
    Args:
    - train_loader: the DataLoader for the training dataset.
    - epochs: the number of epochs to train the model.
    - k: the dimension of the input space.
    - hidden_dim: the dimension of the hidden layer.
    - sigma_w: the standard deviation of the weights initialization.
    - sigma_b: the standard deviation of the bias initialization.
    - device: the device to run the training.
    
    Returns:
    - model: the trained model.
    """
    sigma_w = sigma_w if sigma_w is not None else 1
    sigma_b = sigma_b if sigma_b is not None else 0.0

    model = NTKMLP(k, hidden_dim, k, sigma_w, sigma_b).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.functional.mse_loss

    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        #if epoch % 500 == 0 or epoch == epochs - 1:
        #    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    return model



def train_and_evaluate(k: int, delta: float, outdir: str, trials: int = 1, epochs: int = 2000, batch_size: int = 256, hidden_dim: int = 2048, num_tests: int = 1000, nnz: int = 2, sigma_w: float = None, sigma_b: float = None, max_models: int = 5_000, device: str = 'cpu', seed: int = 0):
    """
    Train a model on a dataset generated with a random permutation matrix P.
    Evaluate the model on a test set generated with the same permutation matrix P.
    Args:
    - k: the dimension of the input space.
    - delta: the desired error.
    - outdir: the directory to save the results.
    - epochs: the number of epochs to train the model.
    - batch_size: the batch size for training.
    - hidden_dim: the dimension of the hidden layer.
    - num_tests: the number of test samples to evaluate the model.
    - nnz: the number of non-zero elements in the test samples.
    - sigma_w: the standard deviation of the weights initialization.
    - sigma_b: the standard deviation of the bias initialization.
    - device: the device to run the training and evaluation.
    
    Returns:
    - num_models: the number of models trained before reaching the desired error delta.
    """
    for t in range(trials):
        os.makedirs(f"{outdir}/{k}/{t}", exist_ok=True)
        torch.manual_seed(seed+t)
        P = dataset.generate_permutation(k, device=device)
        
        train_data = dataset.PermutationDataset(P)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        X_test = dataset.generate_test(k, nnz, num_tests, device)
        Y_test = (P @ X_test.T).T
        X_test = X_test / X_test.norm(p=2, dim=1, keepdim=True)
        Y_test_scaled = Y_test / Y_test.norm(p=2, dim=1, keepdim=True)

        torch.save(P.to("cpu"), f"{outdir}/{k}/{t}/permutation.pt")
        torch.save(X_test.to("cpu"), f"{outdir}/{k}/{t}/test_set_X.pt")
        torch.save(Y_test.to("cpu"), f"{outdir}/{k}/{t}/test_set_Y.pt")

        acc = 0
        error = 1e6
        num_models = 1
        all_preds = None

        while acc < 1 - delta and num_models < max_models:
            model = train_model(train_loader, epochs, k, hidden_dim, sigma_w, sigma_b, device)

            with torch.no_grad():
                pred = model(X_test).unsqueeze(0)
                if all_preds is None:
                    all_preds = pred
                else:
                    all_preds = torch.cat((all_preds, pred), dim=0)

                mean = torch.mean(all_preds, dim=0)
                error = torch.nn.functional.mse_loss(mean, Y_test_scaled)
                error = error if not torch.isnan(error) else 1e6

                mean[mean > 0] = 1.
                mean[mean <= 0] = 0.
                acc = torch.mean((mean.int() == Y_test.int()).all(dim=1).float())

            print(f"Trial {t+1}, Num. models: {num_models} Error: {error:.3f} Acc: {acc:.3f}")
            torch.save(all_preds.to("cpu"), f"{outdir}/{k}/{t}/predictions.pt")

            with open(f"{outdir}/{k}/{t}/results.csv", "a") as f:
                f.write(f"{num_models},{error},{acc}\n")

            num_models += 1

        return num_models


argparse = argparse.ArgumentParser()
argparse.add_argument("--k", type=int, default=10)
argparse.add_argument("--delta", type=float, default=0.1)
argparse.add_argument("--outdir", type=str, default="results")
argparse.add_argument("--trials", type=int, default=10)
argparse.add_argument("--epochs", type=int, default=2500)
argparse.add_argument("--batch_size", type=int, default=256)
argparse.add_argument("--hidden_dim", type=int, default=-1)
argparse.add_argument("--num_tests", type=int, default=1000)
argparse.add_argument("--nnz", type=int, default=2)
argparse.add_argument("--sigma_w", type=float, default=None)
argparse.add_argument("--sigma_b", type=float, default=None)
argparse.add_argument("--max_models", type=int, default=5_000)
argparse.add_argument("--device", type=str, default="cpu")
argparse.add_argument("--seed", type=int, default=0)

if __name__ == "__main__":
    args = argparse.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Using CPU instead.")
        args.device = "cpu"
    args.hidden_dim = args.k*2000 if args.hidden_dim == -1 else args.hidden_dim
    os.makedirs(f"{args.outdir}/{args.k}", exist_ok=True)
    train_and_evaluate(args.k, args.delta, args.outdir, args.trials, args.epochs, args.batch_size, args.hidden_dim, args.num_tests, args.nnz, args.sigma_w, args.sigma_b, args.max_models, args.device, args.seed)
