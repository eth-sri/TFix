# TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer

TFix is a state-of-the-art system for automatically fixing coding errors in programs. The key idea behind TFix is to leverage a large text-to-text Transformer pre-trained on natural languages. This design allows TFix to apply a knowledge transfer between natural and programming languages. In addition to that, TFix is fine-tuned jointly on 52 different error types, which allows it to learn typical patterns across various error types together.

## Setup

To use TFix, you need Python 3. First create a virtual environment and install the dependencies:

```
python3 -m venv venv_tfix
source venv_tfix/bin/activate
pip install -r requirements.txt
```

Note that you may need to install `torch`, `torchvision` and `torchtext` according to your GPU and CUDA version.

## Dataset and Models
The dataset and trained models used in our experiments are available under [this link](https://drive.google.com/file/d/1CtfnYaVf-q6FZP5CUM4Wh7ofpp8b9ajW/view?usp=sharing). Download and unzip them into the same directory as the code. We used the model named `t5large` for TFix.

## Configuration

The main script is `transformers_global.py`. The script has the following arguments:

Required arguments:
```
 -mn --model-name   Name of the model. Choices: [t5-small, t5-base, t5-large, t5-3b, t5-11b]   
```
Optional arguments: 

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| -e --epochs 	            |1              |Number of epochs to fine-tune the model
| -bs  --batch-size         | 1             |Batch size to be used in fine-tuning
| -lr -–learning-rate 	    |1e-4	        |The initial learning rate for fine-tuning
| -gcv --gradient-clip-val  |0.0	        |The maximum allowed norm of the gradient (0.0 means no clipping) 
| -wd --weight-decay 		|0.0            |The strength of adding L2-loss to the fine-tuning loss
| -lm  -–load-model	        |''             |The path to a trained model to run testing 
| -ea --eval-all	        |False          |Whether to evaluate model on <em>random test</em> or not
| -ds --dataset-size        |0              |The chunk size of data during testing.
| -eas --eval-acc-steps     |1              |Number of accumulation samples during evaluation and testing 
| -md --model-dir           |''             |Directory name for the model to save checkpoints and testing results
| -et --error-type			    |''	        |The error type for fine-tuning or testing
| -stl --save-total-limit	|-1             |Maximum number of checkpoints to save
| -pt --pre-trained			|True     	    |Whether to use the pre-training model or not 

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

## An example for testing
```
python transformers_global.py -mn t5-large -lm data_and_models/models/t5large -md t5large_test
```
The testing results consist of two files `first_accs.txt` and `test_data.json`, and are saved in the model directory. `first_accs.txt` reports the exact match accuracy for each error type and `test_data.json` stores the TFix's output fixes in JSON format.

## An example for fine-tuning
```
python transformers_global.py -e 30 -bs 32 -mn t5-large -md t5large_new
```

You can use the `CUDA_VISIBLE_DEVICES` flag to control the GPUs used for fine-tuning or testing.

## Reproducing the experiment results

We provide scripts to obtain the exact match accuracy in our experiments. Obtaining the error removal accuracy involves other complex logics (e.g., calling ESLint) which we plan to release in the future.

### Experiment: Model Size

Obtaining testing results with provided trained models:
```
python transformers_global.py -mn t5-large -lm data_and_models/models/t5large -md t5large_test
python transformers_global.py -mn t5-base -lm data_and_models/models/t5base -md t5base_test
python transformers_global.py -mn t5-small -lm data_and_models/models/t5small -md t5small_test
```

You can also fine-tune the models by yourself:
```
python transformers_global.py -e 30 -bs 32 -mn t5-large -md t5large_new
python transformers_global.py -e 30 -bs 32 -mn t5-base -md t5base_new
python transformers_global.py -e 30 -bs 32 -mn t5-small -md t5small_new
```

### Experiment: No Pre-training

Obtaining testing results with the provided trained model:
```
python transformers_global.py -mn t5-large -lm data_and_models/models/t5large-no-pretrain -md t5large-no-pretrain_test
```

You can also train the model by yourself:
```
python transformers_global.py -e 30 -bs 32 -mn t5-large -md t5large-no-pretrain_new -pt False
```

### Experiment: Fine-tuning per error type
We show how to perform the experiment for the `guard-for-in` error type. For other errors, simply change the `-et` argument. For disk space reasons, we only provide the fine-tuned model for the `guard-for-in` type.

Obtaining testing results with the provided trained model:
```
python transformers_global.py -mn t5-large -lm data_and_models/models/t5large_guard-for-in -et guard-for-in -md t5large_guard-for-in_test
```

You can also fine-tune the model by yourself.
```
python transformers_global.py -e 30 -bs 32 -mn t5-large -md t5large_guard-for-in_new -et guard-for-in
```
