from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_warning_types(data):
    all_warnings = []
    for sample in data:
        if sample.linter_report.rule_id not in all_warnings:
            all_warnings.append(sample.linter_report.rule_id)
    return all_warnings

def filter_rule(data, rule_type):
    filtered_data = []
    for point in data:
        if point.linter_report.rule_id == rule_type:
            filtered_data.append(point)
    return filtered_data

def split_filtered(filtered_data, include_warning, model_name, seed=13):
    filtered_data_temp = filtered_data

    inputs = [data_point.GetT5Representation(include_warning)[0] for data_point in filtered_data]
    outputs = [data_point.GetT5Representation(include_warning)[1] for data_point in filtered_data_temp]

    # if len(inputs) == 1:
    #     # return [], [], [], [], inputs, outputs, [], [], filtered_data
    #     return inputs, outputs, inputs, outputs, inputs, outputs, filtered_data, filtered_data, filtered_data

    test_size = 0.1 if len(inputs) >= 10 else 1 / len(inputs)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, outputs, shuffle=True, random_state=seed, test_size=test_size)

    train_info, test_info = train_test_split(filtered_data, shuffle=True, random_state=seed, test_size=test_size) 

    val_size = 0.1 if len(train_inputs) >= 10 else 1 / len(train_inputs)
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, shuffle=True, random_state=seed, test_size=val_size)
        
    train_info, val_info = train_test_split(train_info, shuffle=True, random_state=seed, test_size=test_size) 

    return train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels, train_info, val_info, test_info

def create_data(data, linter_warnings: list, include_warning, model_name):
    train, train_labels, val, val_labels = [], [], [], []
    test, test_labels = defaultdict(list), defaultdict(list)
    n_test_samples = 0

    train_info, val_info = [], []
    test_info = defaultdict(list)

    for warning in linter_warnings:
        filtered_data = filter_rule(data, warning)        
        train_w, train_w_labels, val_w, val_w_labels, test_w, test_w_labels, train_w_info, val_w_info, test_w_info = split_filtered(filtered_data, include_warning, model_name)

        train += train_w
        train_labels += train_w_labels

        val += val_w
        val_labels += val_w_labels

        train_info += train_w_info
        val_info += val_w_info

        test[warning] = test_w
        test_labels[warning] = test_w_labels

        test_info[warning] = test_w_info

        n_test_samples += len(test_w)
    print('train size: {}\nval size: {}\ntest size: {}'.format(len(train), len(val), n_test_samples))
    return train, train_labels, val, val_labels, test, test_labels, train_info, val_info, test_info

class BugFixDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, targets):
        self.encodings = encodings
        self.target_encodings = targets

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.target_encodings['input_ids'][index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def create_dataset(inputs, labels, tokenizer, pad_truncate, max_length=None):
    if max_length is not None:
        input_encodings = tokenizer(inputs, truncation=pad_truncate, padding=pad_truncate, max_length=max_length)
        # print(type(input_encodings))
        # print(dir(input_encodings))
        # print(input_encodings.input_ids)
        label_encodings = tokenizer(labels, truncation=pad_truncate, padding=pad_truncate, max_length=max_length)
    else:
        input_encodings = tokenizer(inputs, truncation=pad_truncate, padding=pad_truncate, max_length=256)
        label_encodings = tokenizer(labels, truncation=pad_truncate, padding=pad_truncate, max_length=256)

    dataset = BugFixDataset(input_encodings, label_encodings)
    return dataset

def compute_test_results(model, tokenizer, test_inputs, test_labels, test_info, warning_included):
    beam_accuracy = defaultdict(float)
    first_accuracy = defaultdict(float)
    predictions = defaultdict(str)
    # test inputs and labels are dictionaries
    for i, (warning_type, test_data) in enumerate(test_inputs.items()):
        print(f'rule {i}: {warning_type}')
        test_target = test_labels[warning_type] 
        test_w_info = test_info[warning_type]

        predictions_rule, beam_acc, first_acc, test_info_pred = compute_test_results_for_rule(model, tokenizer, test_data, test_target, test_w_info, warning_included)

        test_info[warning_type] = test_info_pred
        predictions[warning_type] = predictions_rule
        beam_accuracy[warning_type] = beam_acc
        first_accuracy[warning_type] = first_acc
    
    return predictions, beam_accuracy, first_accuracy, test_info

def compute_test_results_for_rule(model ,tokenizer, test_inputs, test_labels, test_info, warning_included):
    output = ""
    beam_perfect_counter = 0
    most_probable_perfect_counter = 0

    # here add the predictions into test_info ! return it override in above.
    for i, code in tqdm(list(enumerate(test_inputs))):
        # output += "CODE:\n{}\nFIXED CODE:\n{}\n".format(code, test_labels[i])
        if i % 10000 == 0:
            print("at the index: ", i)
        input_ids = tokenizer.encode(code, return_tensors='pt', truncation=True, padding=True).to(model.device)
        target_max_length = input_ids.shape[1] if warning_included else input_ids.shape[1] + 10
        
        if target_max_length == 1:
            target_max_length = 11
        beam_outputs = model.generate(input_ids, max_length=target_max_length, num_beams=5, early_stopping=False, num_return_sequences=5)

        ref_match_scores = []
        target_ids = tokenizer.encode(test_labels[i], return_tensors='pt', truncation=True, padding=True)
        target_ids = target_ids.squeeze(dim=0).numpy()

        # output += "Outputs:\n"
        perfect_match_in_beam = False
        perfect_match_most_probable = False
        predictions = []
        for j, beam_output in enumerate(beam_outputs):
            ref_match_score = compute_ref_match(beam_output, target_ids)

            if ref_match_score == 1:
                perfect_match_in_beam = True
                if j == 0:
                    perfect_match_most_probable = True

            ref_match_scores.append(ref_match_score)

            pred = tokenizer.decode(beam_output, skip_special_tokens=True)
            predictions.append(pred)
            # output += "{}: {}\n".format(j, pred) 
        
        
        test_info[i].predictions = predictions


        if perfect_match_in_beam:
            beam_perfect_counter += 1
        if perfect_match_most_probable:
            most_probable_perfect_counter += 1
        
        # output += "Perfect Match: {}\nPerfect Match Most Probable: {}\n\n".format(str(perfect_match_in_beam), str(perfect_match_most_probable))
    
    test_size = len(test_inputs)
    beam_accuracy = beam_perfect_counter / test_size
    most_probable_accuracy = most_probable_perfect_counter / test_size
    return output, beam_accuracy, most_probable_accuracy, test_info

def compute_ref_match(output, target_ids):
    target_len = target_ids.shape[0]

    output = output.cpu().numpy()
    output = np.roll(output, -1, axis=0)
    output[-1] = 1
    index_eos = np.argwhere(output == 1)[0][0]
    output = output[:(index_eos+1)]

    if target_len < output.shape[0]:
        diff_length = output.shape[0] - target_len
        ref_match_score = np.sum(target_ids == output[:target_len])
        ref_match_score = ref_match_score / (target_len + diff_length)
    else:
        diff_length = target_len - output.shape[0]
        ref_match_score = np.sum(target_ids[:output.shape[0]] == output)
        ref_match_score = ref_match_score / (target_len + diff_length)
    return ref_match_score
