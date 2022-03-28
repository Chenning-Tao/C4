# C4: Contrastive Cross-Language Code Clone Detection
Code and dataset for paper C4: Contrastive Cross-Language Code Clone Detection

pair_train.jsonl contains the training dataset.

pair_valid.jsonl contains the valid dataset.

pair_test.jsonl contains the test dataset.

The code is in run_con.py. To run the full pipeline, you can enter the following command.

```bash
cd code
output=test 
lr=5e-5
batch_size=36
source_length=512
data_dir=./
output_dir=model/$output
train_file=./pair_train.jsonl
dev_file=./pair_valid.jsonl
test_file=./pair_test.jsonl
epochs=10
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run_con.py --do_train --do_eval --do_test --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --test_filename $test_file
```

To run the above code, we use 3 RTX 3090, average time for each epoch is about 60 min.
