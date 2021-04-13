from ..train import init_without_datasets, load_model, Parser


def get_model(device: str, weights_path: str = './stored_models/model_weights.model') -> Parser:
    print('Initializing model...')
    model = init_without_datasets(device=device)
    print('Initialized.')

    print('Loading pre-trained parameters...')
    load_model(model, load=weights_path, map_location=device)
    print('Loaded.')
    return model
