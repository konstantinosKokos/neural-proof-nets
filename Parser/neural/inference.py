from Parser.train import init_without_datasets, load_model
_weights_path = './stored_models/model_weights.p'


print('Initializing model...')
model = init_without_datasets()
print('Initialized.')

print('Loading pre-trained parameters...')
load_model(model, load=_weights_path)
print('Loaded.')
