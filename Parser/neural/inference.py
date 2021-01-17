from Parser.train import init_without_datasets, load_model, Parser
# _weights_path = './stored_models/model_weights.p'
_weights_path = './stored_models/try/90.model'


def get_model(device: str) -> Parser:
    print('Initializing model...')
    model = init_without_datasets(device=device)
    print('Initialized.')

    print('Loading pre-trained parameters...')
    load_model(model, load=_weights_path, map_location=device)
    print('Loaded.')
    return model
