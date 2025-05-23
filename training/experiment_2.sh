for k in 5 10 15 20 25 30; do
    python experiment.py --k $k --outdir results/experiment_2 --trials 1 --hidden_dim 50000 --sigma_w 1 --seed 2 --device cuda --epochs 10000 --delta 0.001
done