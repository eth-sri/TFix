# TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer

TFix is a state-of-the-art system for automatically fixing coding errors in programs. The key idea behind TFix is to leverage a large text-to-text Transformer pre-trained on natural languages. This design allows TFix to apply a knowledge transfer between natural and programming languages. In addition to that, TFix is fine-tuned jointly on 52 different error types, which allows it to learn typical patterns across various error types together.

<br/>

## Paper

The paper can be found under [this](https://www.semanticscholar.org/paper/TFix%3A-Learning-to-Fix-Coding-Errors-with-a-Berabi-He/0505f17c4052366cbc4fad99150d3542edf85faa) and
[this link](https://files.sri.inf.ethz.ch/website/papers/icml21-tfix.pdf).

If you find our paper useful, please cite:

```
@inproceedings{Berabi2021TFixLT,
  title={TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer},
  author={Berkay Berabi and Jingxuan He and Veselin Raychev and Martin T. Vechev},
  booktitle={ICML},
  year={2021}
}
```

Citation for other formats like `Endnote`, `APA` and a more detailed `BibTeX` citation can be found [here](https://proceedings.mlr.press/v139/berabi21a.html) at the very bottom.

<br/>

## Setup

To use TFix, you need Python 3. First create a virtual environment and install the dependencies:

```
python3 -m venv venv_tfix
source venv_tfix/bin/activate
pip install -r requirements.txt
```

Note that you may need to install `torch`, `torchvision` and `torchtext` according to your GPU and CUDA version.

<br/>

## Linters Setup

We provide our configuration and javascript files to run the ESLint with TFix's configuration and versions. Please note that it is crucial to have the same setup as otherwise the reported bugs can differ with varying configurations and versions.

### ESLint with default config.
```shell
cd linters/javascript/eslint_eslint
npm ci
npx webpack
```
### Eslint with "repo_specific" config
```shell
cd linters/javascript/eslint_repo_specific/
npm ci
npx webpack
```

From now, you should be able to run linters with both configuration. There are two options to run them. You can either directly provide a path to the file or you can provide a line containign a json line with certain fields. The latter can be more useful when you want to run it programatically on many files by using pipes.  Please see the example commands below for more information.

```shell
// running eslint with default config on a file. The commands are the same for the repo_specific config. Just switch the directory accordingly.
cd linters/javascript/eslint_eslint
```

Option 1: Provide a path to the file you want to lint.
```shell
node src/index.js my_file_for_linting.js
```

Option 2: Provide a json as input. When you use this option, the running code will wait for an input from the standard input. So, the code hangs until you provide the input. The code expectsa single line that is a json line.

```shell
node src/index.js -- - JSON
// now provide a json like wit h the following format.
[{"code":"var x = 5;\nvar y = 11;\n","file_name":"some/folder/some_file.js"}]
```

## Dataset and Models

The dataset and trained models used in our experiments are available under [this link](https://drive.google.com/file/d/1CtfnYaVf-q6FZP5CUM4Wh7ofpp8b9ajW/view?usp=sharing). Download and unzip them into the same directory as the code. We used the model named `t5large` for TFix.

The dataset contains a metadata for each data point (buggy-fixed code pair). The metadata is described in the table below.

| Fields          | Description                                                                                                                                            |
| :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| source_code     | The code patch around the error that is fed to the model.                                                                                              |
| target_code     | The corresponding fixed version of the buggy patch. It is fed to the model as target output.                                                           |
| repo            | The repository from which the sample was extracted. Ignore the local path `/data/all/data`                                                             |
| source_changeid | The commit id of the file with the bug                                                                                                                 |
| target_changeid | The commit id of the file with the fix                                                                                                                 |
| source_filename | The name of the file with the bug                                                                                                                      |
| target_filename | The name of the file with the fix                                                                                                                      |
| source_file     | It contains a larger patch around the bug that was used for debugging purposes. The name is misleasind since it does not contain the whole file.       |
| target_file     | It contains a larger patch around the fix that was used for debugging purposes. The name is misleasind since it does not contain the whole file.       |
| linter_report   | Contains the information reported by detector. It hals its own sub-fields like `rule_id` (error type) `message` (error message), `line_begin` and more |
| warning_line    | The line of code on which the errror was reported by the detector.                                                                                     |
| instructions    | It contains a list of text edit operation that explains the diff between `source_code` and `target_code`                                               |

<br/>

## Configuration

The main scripts are `tfix_training.py` and `tfix_testing.py`.

The `tfix_training.py` has the following arguments:

Required arguments:

```
 -mn --model-name   Name of the model. Choices: [t5-small, t5-base, t5-large, t5-3b, t5-11b]
```

Optional arguments:

| Parameter                | Default | Description                                                          |
| :----------------------- | :-----: | :------------------------------------------------------------------- |
| -e --epochs              |    1    | Number of epochs to fine-tune the model                              |
| -bs --batch-size         |    1    | Batch size to be used in fine-tuning                                 |
| -lr -–learning-rate      |  1e-4   | The initial learning rate for fine-tuning                            |
| -gcv --gradient-clip-val |   0.0   | The maximum allowed norm of the gradient (0.0 means no clipping)     |
| -wd --weight-decay       |   0.0   | The strength of adding L2-loss to the fine-tuning loss               |
| -eas --eval-acc-steps    |    1    | Number of accumulation samples during evaluation and testing         |
| -md --model-dir          |   ''    | Directory name for the model to save checkpoints and testing results |
| -et --error-type         |   ''    | The error type for fine-tuning or testing                            |
| -stl --save-total-limit  |   -1    | Maximum number of checkpoints to save                                |
| -pt --pre-trained        |  True   | Whether to use the pre-training model or not                         |

<br/>
<br/>

The `tfix_testing.py` has the following arguments:
Required arguments:

```
 -lm --load-model   Path to the model's checkpoint that will be tested.
```

Optional arguments:

| Parameter             | Default | Description                                                          |
| :-------------------- | :-----: | :------------------------------------------------------------------- |
| -bs --batch-size      |    1    | Batch size to be used in fine-tuning                                 |
| -lm -–load-model      |   ''    | The path to a trained model to run testing                           |
| -ea --eval-all        |  False  | Whether to evaluate model on <em>random test</em> or not             |
| -eas --eval-acc-steps |    1    | Number of accumulation samples during evaluation and testing         |
| -md --model-dir       |   ''    | Directory name for the model to save checkpoints and testing results |
| -et --error-type      |   ''    | The error type for fine-tuning or testing                            |

To fit the program in memory, you should adjust arguments like `-bs` based on your machine configuration. Below you can find detailed information for some of the parameters.

**Parameter -lm**

The code treats the fine-tuning and testing as entirely separate procedures. This means that once the fine-tuning is done, the testing procedure will not start automatically. To test a trained model, you need to use this parameter and pass a valid checkpoint (saved during fine-tuning and can be found in the model's directory). When the `-lm` flag is not empty, the code will automatically switch to testing mode.

**Parameter -ea**

There are two testing dataset: <em>clean test</em> and <em>random test</em>. When the flag `-ea` is set to false, the testing runs only on <em>clean test</em>.
If it is set to true, the testing is done on the large <em>random test</em>.

**Parameter -ds**

This parameter is used during testing and specifies the chunk size of the testing dataset. If this parameter is set, the testing dataset will be split into multiple chunks given the chunk size to fit the program in memory.

**Parameter -eas**

This parameter does not affect testing or fine-tuning but can save you from memory overflows. The testing is done on GPUs, and the samples will be pushed back to the CPU every -eas step to open up space in GPU memory. So you can try to lower this value if you encounter memory overflows during validation and testing.

**Parameter -md**

The model directory used for saving fine-tuning checkpoints or testing results. You can specify a directory name or a default directory name consists of date and time is given. It would help if you use this parameter to name the experiments that you run.

**Parameter -et**

This parameter specifies the error type when fine-tuning or testing per error type.

**Parameter -pt**

Either you can train T5 from scratch, or you can fine-tune it. Setting the flag -pt to true uses pre-trained model.

<br/>

## An example for testing

```
python tfix_testing.py -mn t5-large -lm data_and_models/models/t5large -md t5large_test
```

The testing results consist of two files `first_accs.txt` and `test_data.json`, and are saved in the model directory. `first_accs.txt` reports the exact match accuracy for each error type and `test_data.json` stores the TFix's output fixes in JSON format.

## An example for fine-tuning

```
python tfix_training.py -e 30 -bs 32 -mn t5-large -md t5large_new
```

You can use the `CUDA_VISIBLE_DEVICES` flag to control the GPUs used for fine-tuning or testing.

<br/>

## Reproducing the experiment results

We provide scripts to obtain the exact match accuracy in our experiments. Obtaining the error removal accuracy involves other complex logics (e.g., calling ESLint) which we plan to release in the future.

<br/>

### Experiment: Model Size

Obtaining testing results with provided trained models:

```
python tfix_testing.py -mn t5-large -lm data_and_models/models/t5large -md t5large_test
python tfix_testing.py -mn t5-base -lm data_and_models/models/t5base -md t5base_test
python tfix_testing.py -mn t5-small -lm data_and_models/models/t5small -md t5small_test
```

You can also fine-tune the models by yourself:

```
python tfix_training.py -e 30 -bs 32 -mn t5-large -md t5large_new
python tfix_training.py -e 30 -bs 32 -mn t5-base -md t5base_new
python tfix_training.py -e 30 -bs 32 -mn t5-small -md t5small_new
```

<br/>

### Experiment: No Pre-training

Obtaining testing results with the provided trained model:

```
python tfix_testing.py -mn t5-large -lm data_and_models/models/t5large-no-pretrain -md t5large-no-pretrain_test
```

You can also train the model by yourself:

```
python tfix_training.py -e 30 -bs 32 -mn t5-large -md t5large-no-pretrain_new -pt False
```

<br/>

### Experiment: Fine-tuning per error type

We show how to perform the experiment for the `guard-for-in` error type. For other errors, simply change the `-et` argument. For disk space reasons, we only provide the fine-tuned model for the `guard-for-in` type.

Obtaining testing results with the provided trained model:

```
python tfix_testing.py -mn t5-large -lm data_and_models/models/t5large_guard-for-in -et guard-for-in -md t5large_guard-for-in_test
```

You can also fine-tune the model by yourself.

```
python tfix_training.py -e 30 -bs 32 -mn t5-large -md t5large_guard-for-in_new -et guard-for-in
```
