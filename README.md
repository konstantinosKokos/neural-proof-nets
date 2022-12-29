# neural-roof-nets
## About
Code for the paper Neural Proof Nets ([2009.12702](https://arxiv.org/abs/2009.12702)).

## Update 12/2022
No longer maintained. Follow [spindle](https://github.com/konstantinosKokos/spindle) for updates.

## Update 19/04/2021
This branch contains an updated version of the original submission, with many major improvements
on the output data structures, model architecture and training process.
Please make sure to also update your trained weights when upgrading.
* The bi-modal decoder is replaced with an encoder and a highway connection to the supertagging decoder. This way we can still get 
lexically informed atom representations but without the quadratic memory cost of the cross attention matrix.
* Beam search now penalizes structural errors (like incorrect type constructions and frames failing an invariance check) -- searching with a high beam width
should now return more passing analyses, increasing coverage but at the cost of inference time.
* [RobBert](https://github.com/iPieter/RobBERT) used instead of BERTje. 

The new model's benchmarks and relative differences to the original are reported in the table below.

| Metric (%)       | Greedy     | Beam (2)     |  Beam (3)       | Beam (5)     |  Type Oracle     | 
| :------------- | :----------: | -----------: | :------------- | :----------: |  :----------: |
|  Coverage | 89.9  (N/A)  | 95.3 (N/A)    | 96.1 (N/A) |  97 (N/A)| 97.2 (N/A) |
| Token Accuracy   | 88.5 (+3.0) | 93.0 (+1.6) | 93.3 (+0.9) | 93.7 (+0.5) | -- |
| Frame Accuracy   | 65.7 (+8.1) | 68.1 (+2.8)| 69.1 (+1.1) | 70.0 (+0.4) | -- |
|  λ→ Accuracy  | 69.4 (+9.4) | 71.1 (+5.5) | 71.8 (+4.1) | 72.6 (+3) | 91.2 (+5.8) |
|  λ→◇□ Accuracy | 66.4 (+9.5) | 68.8 (+5.1) | 69.8 (+3.9) | 70.6 (+2.9) | 91.2 (+5.8) |

## Usage

### Installation
Python 3.9+

Clone the project locally. In a clean python venv do `pip install -r requirements.txt`

### Inference
To run the model in inference mode:
1. Download pretrained weights from [here](https://surfdrive.surf.nl/files/index.php/s/BYVKNzD8RPccQhP)
2. Unzip the downloaded file, and place its contents in a directory `stored_models`, alongside `Parser`.
Your resulting directory structure should look like:

    ```
    +--Parser
       +--data
       +--neural
       +--parsing
       +--train.py
       +--__init__.py
    +--stored_models
       +--model_weights.model
    +--README.md
    +--requirements.txt
    ```
3. Run a python console from your working directory, and run:
    ```
    >>> from Parser.neural.inference import get_model
    >>> model = get_model(device)
    >>> analyses = model.infer(xs, n)
    ```
    where `device` is either `"cpu"` or `"cuda"`, `xs` a list of strings to parse and `n` the beam size.
    `analyses` will be a list (one item per input string) of lists (one item per beam) of 
    [Analysis](https://github.com/konstantinosKokos/neural-proof-nets/blob/539036f32373a3e28f7350fb0c5a6f44af7107fc/Parser/parsing/postprocessing.py#L96) objects.
    A non-failing analysis can be converted into a λ-term, as in the example below:
    ```
    >>> sent = "Wat is de lambda-term van dit voorbeeld?"
    >>> analysis = model.infer([sent], 1)[0][0]
    >>> proofnet = analysis.to_proofnet()
    >>> proofnet.print_term(show_words=True, show_types=False, show_decorations=True)
    'Wat ▵ʷʰᵇᵒᵈʸ(λx₀.is ▵ᵖʳᵉᵈᶜ(▿ᵖʳᵉᵈᶜ(x₀)) ▵ˢᵘ(▾ᵐᵒᵈ(van ▵ᵒᵇʲ¹(▾ᵈᵉᵗ(dit) voorbeeld?)) ▾ᵈᵉᵗ(de) lambda-term))'
    ```

#### Evaluation
To evaluate on the test set data, follow steps 1 and 2 of the previous paragraph. You will also need a binary version of 
the processed dataset, placed in the outermost project directory.
 
1. You can download a preprocessed version [here](https://surfdrive.surf.nl/files/index.php/s/7w8EbLx08JEogq4). 
    * Alternatively, you can convert the [original dataset](https://github.com/konstantinosKokos/aethel) into the parser
     format yourself by running the script in `Parser.data.convert_aethel` (additionally requires a local clone of the 
     [extraction code](https://github.com/konstantinosKokos/Lassy-TLG-extraction)).
      
    Your directory structure should look like:

    ```
    +--Parser
       +--data
       +--neural
       +--parsing
       +--train.py
       +--__init__.py
    +--stored_models
       +--model_weights.p
    +--processed.p
    +--README.md
    +--requirements.txt
    ```
2. Open a python console and run 
    ```
    >>> from Parser.neural.evaluation import fill_table
    >>> results = fill_table(bs)
    ``` 
    where `bs` the list of beam sizes (ints) to test with. Note that this runs a single model instead of averaging, so 
    a small variation to the paper reported numbers is to be expected.

#### Training
Follow step 1 of previous paragraph and take a look at `Parser.train`.

#### Help
If you get stuck and require assistance or encounter something unexpected, feel free to
 [get in touch](mailto:k.kogkalidis@uu.nl).
