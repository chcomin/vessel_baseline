### Searching for a retina blood vessel baseline...

To run the codes:

```
python train.py --csv_train ../data/DRIVE/train.csv --save_path exp_name
python generate_results.py --config_file ../experiments/exp_name/config.cfg --dataset DRIVE
python analyze_results.py --path_preds ../results/exp_name --dataset DRIVE
```

The initial code is from the [little wnet](https://github.com/agaldran/lwnet) repository