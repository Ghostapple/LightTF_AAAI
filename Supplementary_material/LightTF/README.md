This is a demo for the ligthweight LightTF model. The structue of the model can be found in the `model` folder. The training and testing code can be found in the `models/LightTF.py` file. The datasets can be found in `data/ETTh1.csv` file. 
For Simplicity, we only provide ETTh1 dataset. 
To run the model, you can use the following command to exeute the `run_longExp.py` file. 
```bash
python run_longExp.py --data_path ETTh1.csv --data ETTh1 --seq_len 672 --pred_len 96 --patch_size 48 --M 1 --K 1 --cut_freq 25
```

We also provide a configuration that only have 4 parameters on the ETTh1 dataset:
```bash
python run_longExp.py --data_path ETTh1.csv --data ETTh1 --seq_len 96 --pred_len 96 --patch_size 48 --M 24 --K 1 --cut_freq 1
```

The final version will be released after the paper is accepted.