# neural-roof-nets
## About
Code for the paper Neural Proof Nets ([2009.12702](https://arxiv.org/abs/2009.12702)).

## Update 01/02/2021
This branch contains an updated version of the original submission, with many major improvements
on the output data structures, model architecture and training process.

The new model's benchmarks and relative differences to the original are reported in the table below.

| Metric (%)       | Greedy     | Beam (2)     |  Beam (3)       | Beam (5)     | 
| :------------- | :----------: | -----------: | :------------- | :----------: | 
|  Coverage | 87.4  (N/A)  | 92.2 (N/A)    | 92.9 (N/A) |  93.66 (N/A)|
| Token Accuracy   | 88.5 (+3.0) | 93.4 (+2.0) | 93.7 (+1.3) | 94.0 (+0.8) |
| Frame Accuracy   | 67.0 (+2.4) | 69.8 (+4.4)| 70.6 (+2.6) | 71.4 (1.8) |
|  λ→ Accuracy  | 67.4 (+7.4) | 69.4 (+3.8) | 70.0 (+2.3) | 70.6 (+1.5) |
|  λ→◇□ Accuracy | 64.3 (+7.4) | 66.6 (+2.9) | 67.3 (+1.4) | 67.9 (+0.8) |

## Usage

### Installation
Python 3.9+

Clone the project locally. In a clean python venv do `pip install -r requirements.txt`

### Inference
To run the model in inference mode:
1. Download pretrained weights from [here](https://surfdrive.surf.nl/files/index.php/s/qYnyHAk3fUjYI8q)
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
