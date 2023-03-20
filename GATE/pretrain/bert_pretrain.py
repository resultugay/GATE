import numpy as np
import torch
import argparse
import pandas as pd
import os
import collections
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import default_data_collator
from transformers import AutoModelForMaskedLM, Trainer

meaninglessCols = ['id', 'entity_id', 'row_id', 'timestamp']

def MLM(dataSerialized, checkpoint, saved_model_path):
    dataset = Dataset.from_pandas(dataSerialized)
    data = DatasetDict({'train': dataset})
    print('data : ', data)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenize_function(examples):
        result = tokenizer(examples['serialized'])
        if tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
        return result

    def group_texts(examples):
        chunk_size = 128
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)] for k, t in
                  concatenated_examples.items()}
        # create a new labels column
        result['labels'] = result['input_ids'].copy()
        return result

    def whole_word_masking_data_collator(features):
        for feature in features:
            word_ids = feature.pop("word_ids")

            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            mask = np.random.binomial(1, wwm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)

    def insert_random_mask(batch):
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {'masked_' + k: v.numpy() for k, v in masked_inputs.items()}



    tokenized_datasets = data.map(tokenize_function, batched=True, remove_columns=['serialized'])
    print(tokenized_datasets)
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    downsampled_dataset = lm_datasets['train'].train_test_split(train_size=0.6, test_size=0.4, seed=42)

    batch_size = 32
    logging_steps = len(downsampled_dataset['train']) // batch_size
    training_args = TrainingArguments(output_dir=saved_model_path,
                                      overwrite_output_dir=True,
                                      evaluation_strategy='epoch',
                                      learning_rate=2e-5,
                                      weight_decay=0.01,
                                      num_train_epochs=10,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      push_to_hub=False,
                                      fp16=True,
                                      save_total_limit=2,
                                      logging_steps=logging_steps)

    wwm_probability = 0.2
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    trainer = Trainer(model=model, args=training_args, train_dataset=downsampled_dataset['train'],
                      eval_dataset=downsampled_dataset['test'], data_collator=data_collator)

    downsampled_dataset = downsampled_dataset.remove_columns(['word_ids'])
    eval_dataset = downsampled_dataset['test'].map(
        insert_random_mask,
        batched=True,
        remove_columns=downsampled_dataset['test'].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels"
        }
    )

    # before train, first evaluate
    eval_results = trainer.evaluate()
    print('Before training : ', eval_results)

    # train
    trainer.train()

    eval_results = trainer.evaluate()
    print('After training : ', eval_results)

    # save checkpoint
    trainer.save_model()


def main():
    parser = argparse.ArgumentParser(description='MLM')

    parser.add_argument('-data_dir', '--data_dir', type=str, default='../../data/nba')
    parser.add_argument('-filename', '--filename', type=str, default='data')
    parser.add_argument('-model_checkpoint', '--model_checkpoint', type=str, default='distilbert-base-uncased')
    parser.add_argument('-gpu', '--gpu', type=str, default='0')
    parser.add_argument('-saved_model', '--saved_model', type=str, default='../../data/nba/nba-mlm')

    args = parser.parse_args()
    arg_dict = args.__dict__

    def serialize(row, cols):
        context = ''
        for c in cols:
            if c not in meaninglessCols:
                context += '[COL] ' + str(c) + ' [VAL] ' + str(row[c]) + ' '
        return context

    data = pd.read_csv(os.path.join(arg_dict['data_dir'], arg_dict['filename'] + '.csv'))
    data['serialized'] = data.apply(serialize, axis=1, args=([data.columns]))
    print(data)
    print(data[['serialized']])
    # MLM learning
    os.environ['CUDA_VISIBLE_DEVICES'] = arg_dict['gpu']
    os.environ['WANDB_DISABLED'] = "true"
    MLM(data[['serialized']], arg_dict['model_checkpoint'], arg_dict['saved_model'])

if __name__ == '__main__':
    main()