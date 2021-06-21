import torch
import numpy as np
import argparse
import os
import sys
import json
sys.path.append('..')
from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer, T5Config
from data_reader import GetDataAsPython
from prepare_data import create_data, create_dataset, extract_warning_types, compute_test_results, filter_rule, compute_ref_match
from datetime import datetime
from collections import defaultdict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import transformers
from tqdm import tqdm

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("start time:", current_time)

# transformers.logging.set_verbosity_info()
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-bs', '--batch-size', type=int, default=1)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
parser.add_argument('-gcv', '--gradient-clip-val', type=float, default=0.0)
parser.add_argument('-wd', '--weight-decay', type=float, default=0)
parser.add_argument('-mn', '--model-name', type=str, choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'], required=True)
parser.add_argument('-lm', '--load-model', type=str, default='') #  Checkpoint dir to load the model. Example: t5-small_global_14-12-2020_16-29-22/checkpoint-10
parser.add_argument('-ea', '--eval-all', type=boolean_string, default=False) # to evaluate on all data or not
parser.add_argument('-ds', '--dataset-size', type=int, default=0)
parser.add_argument('-eas', '--eval-acc-steps', type=int, default=1)
parser.add_argument('-md', '--model-dir', type=str, default='')
parser.add_argument('-et', '--error-type', type=str, default='')
parser.add_argument('-stl', '--save-total-limit', type=int, default=-1)
parser.add_argument('-pt', '--pre-trained', type=boolean_string, default=True)
args = parser.parse_args()

def average_of_dict(dict):
    total = 0
    N = 0
    for key, value in dict.items():
        total += value
        N += 1
    return total / N

# Create model directory
model_name = args.model_name
if args.model_dir != '':
    model_directory = args.model_dir
else:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_directory = 't5global' + '_' + dt_string
    model_directory = model_name + '_global_' + dt_string

os.makedirs(model_directory)
with open(os.path.join(model_directory, 'commandline_args.txt'), 'w') as f:
    f.write('\n'.join(sys.argv[1:]))

# Read data
data = GetDataAsPython('data_and_models/data/data_autofix_tracking_repo_specific_final.json')
data_eslint = GetDataAsPython('data_and_models/data/data_autofix_tracking_eslint_final.json')
data += data_eslint
all_warning_types = extract_warning_types(data)
if args.error_type != '':
    all_warning_types = [args.error_type]
print(all_warning_types)
train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info = create_data(data, all_warning_types, include_warning=True, model_name=model_name)

# Create model and tokenizer (required in both training and testing)
if args.load_model != '':
    tokenizer = T5Tokenizer.from_pretrained(args.load_model)
    print("Loaded tokenizer from directory {}".format(args.load_model))

    model = T5ForConditionalGeneration.from_pretrained(args.load_model)
    print("Loaded model from directory {}".format(args.load_model))
    # model.parallelize()
    model.to(f'cuda:{torch.cuda.current_device()}')
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer = T5Tokenizer.from_pretrained(model_name,)
    tokenizer.add_tokens(['{', '}', '>', '\\', '^'])
    tokenizer.save_pretrained(model_directory)

    if args.pre_trained:
        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=False)
    else:
        print('training from scratch')
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
    model.parallelize()
    model.resize_token_embeddings(len(tokenizer))
    print("models parameters: ", model.num_parameters())
# Create dataset required by pytorch
train_dataset = create_dataset(train_inputs, train_labels, tokenizer, pad_truncate=True, max_length=128)
val_dataset = create_dataset(val_inputs, val_labels, tokenizer, pad_truncate=True)
# Training args (required by both training and test)
training_args = Seq2SeqTrainingArguments(
        output_dir=model_directory,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=args.weight_decay,
        logging_dir=model_directory,
        logging_steps=100,
        do_eval=True,
        evaluation_strategy='epoch',
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=args.epochs if args.save_total_limit == -1 else args.save_total_limit,
        eval_accumulation_steps=args.eval_acc_steps, # set this lower, if testing or validation crashes
        # disable_tqdm=True if args.load_model != '' else False,
        disable_tqdm=False,
        predict_with_generate=True, # never set this to false, it is for testing.
        # lr_scheduler_type="polynomial",
    )
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=args.learning_rate), None],
    tokenizer=tokenizer,
)


if args.load_model != '':
    print("Testing...")
    if args.eval_all:
        counter = 0
        for key in test_inputs:
            counter += len(test_inputs[key])
        print("size before addition of filtered samples: ", counter)

        file_paths = ["data_and_models/data/data_autofix_tracking_repo_specific_filtered.json", "data_and_models/data/data_autofix_tracking_eslint_filtered.json"]

        for path in file_paths:
            filtered_data = GetDataAsPython(path)
            print("size of filtered data: ", len(filtered_data))

            filtered_test, filtered_test_labels = defaultdict(list), defaultdict(list)
            filtered_test_info = defaultdict(list)
            filtered_warning_types = extract_warning_types(filtered_data)
            if args.error_type != '':
                filtered_warning_types = [args.error_type]

            for warning in filtered_warning_types:
                filtered_test_data = filter_rule(filtered_data, warning)
                filtered_data_temp = filtered_test_data

                filtered_test_inputs = [data_point.GetT5Representation(True)[0] for data_point in filtered_test_data]
                filtered_test_outputs = [data_point.GetT5Representation(True)[1] for data_point in filtered_data_temp]

                filtered_test[warning] = filtered_test_inputs
                filtered_test_labels[warning] = filtered_test_outputs

                filtered_test_info[warning] = filtered_test_data

            for warning in filtered_warning_types:
                test_inputs[warning] = test_inputs[warning] + filtered_test[warning]
                test_labels[warning] = test_labels[warning] + filtered_test_labels[warning]
                test_info[warning] = test_info[warning] + filtered_test_info[warning]

    counter = 0
    for key in test_inputs:
        counter += len(test_inputs[key])
    print("number of testing samples: ", counter)

    # test that the samples are well aligned among inputs and info
    for warning in test_inputs:
        inputs = test_inputs[warning]
        infos = test_info[warning]
        for i, code in enumerate(inputs):
            assert code == infos[i].GetT5Representation(True)[0], "something wrong! stop it!"

    # sorted_warning_types = all_warning_types[:]
    # sorted_warning_types = list(sorted(sorted_warning_types, key=lambda w: len(test_inputs[w]), reverse=True))
    # for w in sorted_warning_types:
    #     print(w, len(test_inputs[w]))

    # Testing
    scores = defaultdict(float)
    for i, warning in enumerate(all_warning_types):
        test_warning = test_inputs[warning]
        test_warning_labels = test_labels[warning]
        test_warning_info = test_info[warning]
        target_max_length = 256 # Set this to 256 if enough memory

        if args.dataset_size == 0:
            ds_size = len(test_warning)
        else:
            ds_size = args.dataset_size

        num_ds_batches = len(test_warning) // ds_size
        if len(test_warning) % ds_size:
            num_ds_batches += 1

        print(f'rule {i}: {warning}, # {len(test_warning)}, #num_ds_batches {num_ds_batches}')
        correct_counter, total_counter = 0, 0
        for j in range(num_ds_batches):
            test_warning_batch = test_warning[j*ds_size:(j+1)*ds_size]
            test_warning_labels_batch = test_warning_labels[j*ds_size:(j+1)*ds_size]
            test_warning_info_batch = test_warning_info[j*ds_size:(j+1)*ds_size]
            test_warning_dataset = create_dataset(test_warning_batch, test_warning_labels_batch, tokenizer, pad_truncate=True, max_length=target_max_length)
            # here use model.generate with batches
            target_ids = tokenizer(test_warning_labels_batch, return_tensors='pt', truncation=True, padding='max_length', max_length=target_max_length).input_ids
            target_ids = np.array(target_ids)

            output_ids = trainer.predict(test_dataset=test_warning_dataset, num_beams=5, max_length=target_max_length).predictions
            output_ids = np.pad(output_ids, ((0,0), (0, target_max_length - output_ids.shape[1])), mode='constant')
            output_ids = np.delete(output_ids, 0, axis=1)
            output_ids = np.insert(output_ids, target_max_length-1, 0, axis=1)

            correct_counter += np.sum(np.all(np.equal(target_ids, output_ids), axis=1))
            total_counter += len(output_ids)
            for k, output_id in enumerate(output_ids):
                pred = tokenizer.decode(output_id, skip_special_tokens=True)
                predictions = []
                predictions.append(pred)
                test_warning_info[j*ds_size+k].predictions = predictions

        scores[warning] = correct_counter / total_counter
        test_info[warning] = test_warning_info
        print(f'rule {i} acc: {correct_counter / total_counter}')
    scores['average'] = average_of_dict(scores)

    # create the whole test list
    test_list = []
    for key in test_info:
        test_list += test_info[key]
    with open(os.path.join(model_directory, 'test_data.json'), 'w') as outfile:
        json.dump([datapoint.__dict__ for datapoint in test_list], outfile, default=lambda o: o.__dict__)

    serialized_scores = json.dumps(scores, indent=4)
    output_file = open(os.path.join(model_directory, 'first_accs.txt'), 'w+')
    output_file.write(serialized_scores)
    output_file.close()
else:
    trainer.train()
    trainer.save_model()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("end time:", current_time)

# no-undef 21039
# no-unused-vars 20015
# no-empty 14430
# no-unreachable 11915
# no-cond-assign 11329
# no-invalid-this 11225
# no-redeclare 10564
# no-console 10405
# no-process-exit 9945
# no-extra-semi 9243
# comma-style 8010
# no-useless-escape 7776
# guard-for-in 7652
# prefer-rest-params 7533
# no-constant-condition 5735
# no-dupe-keys 5727
# no-case-declarations 5502
# no-array-constructor 5478
# no-throw-literal 5111
# generator-star-spacing 5083
# prefer-spread 4993
# no-fallthrough 4858
# no-extra-boolean-cast 4766
# no-extend-native 4227
# no-inner-declarations 3416
# no-debugger 3387
# no-extra-bind 3361
# no-func-assign 3260
# no-self-assign 2394
# no-new-object 1613
# no-this-before-super 1512
# require-yield 1426
# no-caller 1296
# no-const-assign 1227
# no-unsafe-finally 1171
# no-new-wrappers 961
# no-sparse-arrays 751
# no-global-assign 714
# getter-return 681
# no-unsafe-negation 602
# valid-typeof 474
# no-dupe-class-members 415
# constructor-super 397
# no-duplicate-case 303
# no-unused-labels 273
# no-empty-pattern 239
# no-class-assign 235
# no-ex-assign 143
# use-isnan 142
# no-compare-neg-zero 59
# for-direction 36
# no-new-symbol 5