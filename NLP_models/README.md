## Instructions

In this part, we trained three NLP models (BERT, LSTM, and SVM). Each model is trained under three datasets(Liushaung's, Zimmeck's and our PrivAud-100) separately and located in one independent folder.
The name for each folder is `./[model]/[model]_[data_set_name]_dataset`.

## Train and Test
The code is easy for duplication. No need to change any setting in the code.

Run the following command to start training:
```
python3 ./train.py
```
The model file ends with `.ckpt` which is located in:
```
./[model]/[model]_[dataset]_dataset/logs/[model_name]/[version_num]/checkpoints/[arguments].ckpt

```
Run the following command to test the model:
```
python3 ./infer_by_out_corpus.py
```
The result will be a csv file with sentences and predicted results.
```
./policy_all_in_one_filter_purged.csv
```