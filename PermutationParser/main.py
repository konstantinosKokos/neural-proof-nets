from PermutationParser.neural.model import *
from PermutationParser.neural.utils import *
from PermutationParser.neural.schedules import *
from PermutationParser.data.sample import load_stored

from torch.nn import KLDivLoss

import subprocess
import os
import sys

num_epochs = 1000
warmup_epochs = 5
restart_epochs = 100
max_lr = 1e-04
linking_weight = 1


def logprint(x: str, ostream: Any) -> None:
    print(x)
    ostream.write(x + '\n')
    sys.stdout.flush()


def load_model(parser: Parser, load: str, **kwargs) -> Tuple[int, Dict, int]:
    print('Loading model parameters...')
    temp = torch.load(load, **kwargs)
    parser.load_state_dict(temp['model_state_dict'])
    step_num = temp['step']
    opt_state_dict = temp['opt_state_dict']
    epoch = temp['epoch'] + 1
    return step_num, opt_state_dict, epoch


def init(datapath: Optional[str] = None, max_len: int = 95, train_batch: int = 64,
         val_batch: int = 512, device: str = 'cuda', version: Optional[str] = None,
         save_to_dir: Optional[str] = None) \
        -> Tuple[DataLoader, DataLoader, DataLoader, int, Parser, str]:
    if version is None:
        version = subprocess.check_output(['git', 'describe', '--always'], cwd='./PermutationParser').strip().decode()
    if save_to_dir is not None:
        os.makedirs(f'{save_to_dir}/', exist_ok=True)
    else:
        os.makedirs(f'./stored_models/{version}/', exist_ok=True)
    print(f'Version id:\t{version}')

    # load data and model
    trainset, devset, testset = load_stored() if datapath is None else load_stored(datapath)

    devset = sorted(devset, key=lambda sample: len(sample.polish))
    testset = sorted(testset, key=lambda sample: len(sample.polish))

    train_dl = make_dataloader([sample for sample in trainset if len(sample.polish) <= max_len], train_batch)
    dev_dl = make_dataloader([sample for sample in devset if len(sample.polish) <= max_len], val_batch, shuffle=False)
    test_dl = make_dataloader(testset, val_batch, shuffle=False)
    nbatches = get_nbatches(max_len, trainset, train_batch)
    print('Read data.')
    parser = Parser(AtomTokenizer(trainset + devset + testset), Tokenizer(), 768, 128, device)
    print('Initialized model.')
    return train_dl, dev_dl, test_dl, nbatches, parser, version


def init_pere(datapath: str, save_to_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader, int, Parser, str]:
    return init(datapath, 95, 256, 512, 'cuda', version='pere', save_to_dir=save_to_dir)


def train(model_path: Optional[str] = None, data_path: Optional[str] = None, pere: bool = False,
          version: Optional[str] = None, save_to_dir: Optional[str] = None):

    if pere:
        train_dl, val_dl, test_dl, nbatches, parser, version = init_pere(data_path, save_to_dir)
    else:
        train_dl, val_dl, test_dl, nbatches, parser, version = init(data_path, version=version, save_to_dir=save_to_dir)

    schedule = make_cosine_schedule_with_restarts(max_lr=max_lr, warmup_steps=warmup_epochs * nbatches,
                                                  restart_every=restart_epochs * nbatches,
                                                  decay_over=num_epochs * nbatches)

    param_groups, grad_scales = list(zip(*[({'params': parser.word_encoder.parameters()}, 0.1),
                                           ({'params': parser.atom_embedder.parameters()}, 1),
                                           ({'params': parser.atom_decoder.parameters()}, 1),
                                           ({'params': parser.atom_encoder.parameters()}, 1),
                                           ({'params': parser.negative_transformation.parameters()}, 1)]))

    _opt = torch.optim.AdamW(param_groups, lr=1e10, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-05)
    opt = Scheduler(_opt, schedule, grad_scales)
    fuzzy_loss = FuzzyLoss(KLDivLoss(reduction='batchmean'), len(parser.atom_tokenizer) + 1, 0.1)

    if model_path is not None:
        step_num, opt_dict, init_epoch = load_model(parser, model_path)
        opt.step_num = step_num
        opt.lr = opt.schedule(opt.step_num)
        opt.opt.load_state_dict(opt_dict)
    else:
        init_epoch = 0

    if save_to_dir is None:
        save_to_dir = './stored_models'

    log = []
    for e in range(init_epoch, num_epochs):
        # epoch settings
        validate = True if e % 20 == 0 and e != 0 else False
        save = True if e % 20 == 0 and e != 0 else True if e == num_epochs - 1 else False
        epoch_lr = opt.lr

        with open(f'{save_to_dir}/{version}/log.txt', 'a') as stream:
            logprint('=' * 64, stream)
            logprint(f'Epoch {e}', stream)
            logprint(' ' * 50 + f'LR: {epoch_lr}', stream)
            logprint(' ' * 50 + f'LW: {linking_weight}', stream)
            logprint('-' * 64, stream)
            supertagging_loss, linking_loss = parser.train_epoch(train_dl, fuzzy_loss, opt, linking_weight)
            logprint(f' Supertagging Loss:\t\t{supertagging_loss:5.2f}', stream)
            logprint(f' Linking Loss:\t\t\t{linking_loss:5.2f}', stream)
            if validate:
                logprint('-' * 64, stream)
                sentence_ac, atom_ac, link_ac = parser.eval_epoch(val_dl, oracle=False, link=linking_weight != 0)
                logprint(f' Sentence Accuracy:\t\t{(sentence_ac * 100):6.2f}', stream)
                logprint(f' Atom Accuracy:\t\t\t{(atom_ac * 100):6.2f}', stream)
                logprint(f' Link Accuracy:\t\t\t{(link_ac * 100):6.2f}', stream)
            logprint('\n', stream)
            log.append((e, epoch_lr, supertagging_loss, linking_loss))

            if save:
                print('\tSaving')
                torch.save({'model_state_dict': parser.state_dict(),
                            'opt_state_dict': opt.opt.state_dict(),
                            'step': opt.step_num,
                            'epoch': e},
                           f'{save_to_dir}/{version}/{e}.model')
