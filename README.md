### Searching for a retina blood vessel baseline...

To run the codes:

```
python train.py --csv_train ../data/DRIVE/train.csv --save_path exp_name
python generate_results.py --config_file ../experiments/exp_name/config.cfg --dataset DRIVE
python analyze_results.py --path_train_preds ../results/DRIVE/experiments/exp_name --path_test_preds ../results/DRIVE/experiments/exp_name --train_dataset DRIVE --test_dataset DRIVE
```

The initial code is from the [little wnet](https://github.com/agaldran/lwnet) repository