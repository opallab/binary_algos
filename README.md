# Learning to Add, Multiply, and Execute Algorithmic Instructions Exactly with Neural Networks

This repository provides empirical validations for the theoretical results presented in the paper “Learning to Add, Multiply, and Execute Algorithmic Instructions Exactly with Neural Networks.”


To run the experiments described in the paper, first install the necessary dependencies by executing:

```bash
conda env create --file=environment.yml
conda activate exact_learning
```

### Results in Appendix B

To reproduce the numerical validations of the constructive proofs in Appendix B using the NTK predictor for binary permutations, additions, and multiplications, refer to the following scripts:
- `validation/validate_permutation.py`: validates the solution for permutation for all numbers up to 10 bits
- `validation/validate_addition.py`: validates the solution for addition for all summands up to 10 bits
- `validation/validate_multiplication.py`: validates the solution for multiplication for all multipliers/multiplicands up to 10 bits 

Make sure that these scripts are called from within the `validation` folder. Since the validation is quite time-intensive, a demonstration of algorithmic execution for all three applications is also provided in the notebook: `validation/demo.ipynb`

## Results in Appendix E
To run the ensemble complexity experiments for learning binary permutations as described in Appendix E, use the following scripts:
- `training/experiment_1.sh`: Generates the results for the left plot in Figure 7.
- `training/experiment_2.sh`: Generates the results for the right plot in Figure 7.

Make sure that these scripts are called from within the `training` folder and that they are given execution permissions, e.g. by invoking `chmod +x experiment_1.sh` and `chmod +x experiment_2.sh`.
