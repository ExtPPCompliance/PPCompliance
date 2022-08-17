## Instructions
For privacy analyzer, it gets the sentence type (21 labels instroduced before)that predicted by our trained BERT model.
The code is a large scale prediction for all the privacy policy sentences of all the extensions.

### Input File
The input file is a large `csv` file with the structure of `[ext_id, sentence]` in each line.

The input file can be found at:
`./implementation_data/all_privacy_policy.csv`

### Run the Code
Very simple, just type in the terminal:
`python3 ./large_scale_infer.py`

### Output Result
The result is also a `csv` file with the structure of `[ext_id, sentence, top1_pred, top2_pred]` in each line.
The result will record the top 2 predictions for any possible analysis.
The output file would be like:
`./need_to_add.csv`
