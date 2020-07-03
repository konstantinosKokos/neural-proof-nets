from Parser.train import init, load_model
_weights_path = './stored_models/model_weights.p'


print('Initializing model...')
model = init(version='3-1-8-256-32-nll')[4]
print('Initialized.')

print('Loading pre-trained parameters...')
load_model(model, load=_weights_path)
print('Loaded.')
