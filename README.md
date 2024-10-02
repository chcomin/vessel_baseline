### Searching for a retina blood vessel baseline...

To run the codes:

```
python train_cyclical.py --csv_train data/DRIVE/train.csv --save_path unet_drive
python generate_results.py --config_file experiments/unet_drive/config.cfg --dataset DRIVE --device cuda:0
python analyze_results.py --path_train_preds results/DRIVE/experiments/unet_drive --path_test_preds results/DRIVE/experiments/unet_drive --train_dataset DRIVE --test_dataset DRIVE
```

The initial code is from the [little wnet](https://github.com/agaldran/lwnet) repository