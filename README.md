It is recomended to use anaconda to install requirments:

```bash
conda env create -f environment.yml
```

In order to test pretrained models you must provide **relative** path to csv file with. Example:
```
python3 lstm.py data/eval.csv
```

