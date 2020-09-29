# neural-roof-nets
## About
Code for the paper Neural Proof Nets ([2009.12702](https://arxiv.org/abs/2009.12702)).

## Usage

### Installation
Python 3.8+

Clone the project locally. In a clean python venv do `pip install -r requirements.txt`

### Inference
To run the model in inference mode:
1. Download pretrained weights from [here](https://surfdrive.surf.nl/files/index.php/s/Af9P4PsZ1qEv04N)
2. Unzip the downloaded file, and place its contents in a directory `stored_models`, alongside `Parser`.
Your resulting directory structure should look like:

    ```
    +---Parser
        +--data
        +--neural
        +--parsing
        +--train.py
        +--__init__.py
    +---stored_models
        +--model_weights.p
    +--README.md
    +--requirements.txt
    ```
3. Run a python console from your working directory, and run:
    ```
    from Parser.neural.inference import get_model
    model = get_model(device)
    analyses = model.infer(xs, n)
    ```
    where `device` is either `"cpu"` or `"cuda"`, `xs` a list of strings to parse and `n` the beam size.
    For manipulating analyses, refer to the `Analysis` class in `Parser.parsing.utils`

#### Evaluation
To evaluate on the test set data, follow steps 1 and 2 of the previous paragraph. You will also need a binary version of 
the processed dataset, placed in the outermost project directory.
 
1. You can download a preprocessed version [here](https://surfdrive.surf.nl/files/index.php/s/7w8EbLx08JEogq4). 
    * Alternatively, you can convert the [original dataset](https://github.com/konstantinosKokos/aethel) into the parser
     format yourself by running the script in `Parser.data.convert_aethel` (additionally requires a local clone of the 
     [extraction code](https://github.com/konstantinosKokos/Lassy-TLG-extraction)).
      
    Your directory structure should look like:

    ```
    +---Parser
        +--data
        +--neural
        +--parsing
        +--train.py
        +--__init__.py
    +---stored_models
        +--model_weights.p
    +---processed.p
    +--README.md
    +--requirements.txt
    ```
2. Open a python console and run 
    ```
    from Parser.neural.evaluation import fill_table
    results = fill_table(bs)
    ``` 
    where `bs` the list of beam sizes (ints) to test with. Note that this runs a single model instead of averaging, so 
    a small variation to the paper reported numbers is to be expected.

#### Training
Follow step 1 of previous paragraph and take a look at `Parser.train`.

#### Help
If you get stuck and require assistance or encounter something unexpected, feel free to
 [get in touch](mailto:k.kogkalidis@uu.nl).