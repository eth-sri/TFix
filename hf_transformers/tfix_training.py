from datetime import datetime
import argparse
import os
import sys

sys.path.append("..")

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch

from data_reader import GetDataAsPython
from prepare_data import create_data
from prepare_data import create_dataset
from prepare_data import extract_warning_types
from utils import boolean_string
from utils import get_current_time

# transformers.logging.set_verbosity_info()
print("start time: ", get_current_time())

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-bs", "--batch-size", type=int, default=1)
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-gcv", "--gradient-clip-val", type=float, default=0.0)
parser.add_argument("-wd", "--weight-decay", type=float, default=0)
parser.add_argument(
    "-mn",
    "--model-name",
    type=str,
    choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
    required=True,
)
parser.add_argument(
    "-lm", "--load-model", type=str, default=""
)  #  Checkpoint dir to load the model. Example: t5-small_global_14-12-2020_16-29-22/checkpoint-10
parser.add_argument(
    "-ea", "--eval-all", type=boolean_string, default=False
)  # to evaluate on all data or not
parser.add_argument("-eas", "--eval-acc-steps", type=int, default=1)
parser.add_argument("-md", "--model-dir", type=str, default="")
parser.add_argument("-et", "--error-type", type=str, default="")
parser.add_argument("-stl", "--save-total-limit", type=int, default=-1)
parser.add_argument("-pt", "--pre-trained", type=boolean_string, default=True)
args = parser.parse_args()

# Create model directory
model_name = args.model_name
if args.model_dir != "":
    model_directory = args.model_dir
else:
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    model_directory = "t5global" + "_" + dt_string
    model_directory = model_name + "_global_" + dt_string

os.makedirs(model_directory)
with open(os.path.join(model_directory, "commandline_args.txt"), "w") as f:
    f.write("\n".join(sys.argv[1:]))

# Read data
data = GetDataAsPython("data_and_models/data/data_autofix_tracking_repo_specific_final.json")
data_eslint = GetDataAsPython("data_and_models/data/data_autofix_tracking_eslint_final.json")
data += data_eslint
all_warning_types = extract_warning_types(data)
if args.error_type != "":
    all_warning_types = [args.error_type]
print(all_warning_types)
(
    train_inputs,
    train_labels,
    val_inputs,
    val_labels,
    test_inputs,
    test_labels,
    train_info,
    val_info,
    test_info,
) = create_data(data, all_warning_types, include_warning=True, model_name=model_name)

tokenizer = T5Tokenizer.from_pretrained(
    model_name,
)
tokenizer.add_tokens(["{", "}", ">", "\\", "^"])
tokenizer.save_pretrained(model_directory)

if args.pre_trained:
    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=False)
else:
    print("training from scratch")
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration(config)
model.parallelize()
model.resize_token_embeddings(len(tokenizer))
print("models parameters: ", model.num_parameters())
# Create dataset required by pytorch
train_dataset = create_dataset(
    train_inputs, train_labels, tokenizer, pad_truncate=True, max_length=128
)
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
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=args.epochs if args.save_total_limit == -1 else args.save_total_limit,
    eval_accumulation_steps=args.eval_acc_steps,  # set this lower, if testing or validation crashes
    disable_tqdm=False,
    predict_with_generate=True,  # never set this to false, it is for testing.
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=[torch.optim.Adam(params=model.parameters(), lr=args.learning_rate), None],
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model()
print("end time: ", get_current_time())
