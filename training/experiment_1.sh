for k in {2..30}; do
    python experiment.py --k $k --outdir results/experiment_1 --trials 1 --hidden_dim 50000 --sigma_w 1 --seed 2 --device cuda --epochs 10000
done