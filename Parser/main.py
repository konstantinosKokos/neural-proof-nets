from Parser.neural.model import *
from Parser.neural.utils import *
from Parser.neural.schedules import *
from Parser.data.sample import load_stored

from torch.nn import KLDivLoss

import subprocess
import os
import sys

num_epochs = 505
warmup_epochs = 5


def logprint(x: str, ostreams: List[Any]) -> None:
    print(x)
    for ostream in ostreams:
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


def init(datapath: Optional[str] = None, max_len: int = 100, train_batch: int = 32,
         val_batch: int = 512, device: str = 'cuda', version: Optional[str] = None,
         save_to_dir: Optional[str] = None) \
        -> Tuple[DataLoader, DataLoader, DataLoader, int, Parser, str]:
    if version is None:
        version = subprocess.check_output(['git', 'describe', '--always'], cwd='./Parser').strip().decode()
    if save_to_dir is not None:
        os.makedirs(f'{save_to_dir}/', exist_ok=True)
    else:
        os.makedirs(f'./stored_models/{version}/', exist_ok=True)
    print(f'Version id:\t{version}')

    # load data and model
    _trainset, _devset, _testset = load_stored() if datapath is None else load_stored(datapath)

    atokenizer = AtomTokenizer(_trainset+_devset+_testset)
    tokenizer = Tokenizer()

    print('Making dataloaders.')
    trainset, devset, testset = list(map(lambda dset: list(filter(lambda sample: len(sample.polish) <= max_len, dset)),
                                         [_trainset, _devset, _testset]))
    nbatches = get_nbatches(trainset, train_batch)
    trainset = [vectorize_sample(s, atokenizer, tokenizer) for s in trainset]
    devset = [vectorize_sample(s, atokenizer, tokenizer) for s in sorted(devset, key=lambda x: len(x.polish))]
    testset = [vectorize_sample(s, atokenizer, tokenizer) for s in sorted(testset, key=lambda x: len(x.polish))]

    train_dl = make_dataloader(trainset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, True,
                               train_batch, True)
    dev_dl = make_dataloader(devset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, False, val_batch, False)
    test_dl = make_dataloader(testset, atokenizer.pad_token_id, tokenizer.core.pad_token_id, 32, False,
                              val_batch, False)

    print('Read data.')
    parser = Parser(atokenizer, tokenizer, device=device)
    print('Initialized model.')
    return train_dl, dev_dl, test_dl, nbatches, parser, version


def train(model_path: Optional[str] = None, data_path: Optional[str] = None,
          version: Optional[str] = None, save_to_dir: Optional[str] = None):

    train_dl, val_dl, test_dl, nbatches, parser, version = init(data_path, version=version, save_to_dir=save_to_dir)

    schedule = make_noam_scheme(d_model=parser.dec_dim, warmup_steps=warmup_epochs * nbatches, factor=1.)
    param_groups, grad_scales = list(zip(*[({'params': parser.word_encoder.parameters(),
                                             'weight_decay': 0}, 0.1),
                                           ({'params': parser.pos_transformation.parameters()}, 1),
                                           ({'params': parser.neg_transformation.parameters()}, 1),
                                           ({'params': parser.linker.parameters()}, 1),
                                           ({'params': parser.atom_embedder.parameters()}, 1),
                                           ({'params': parser.supertagger.parameters()}, 1)]))

    _opt = torch.optim.AdamW(param_groups, lr=1e10, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-05)
    opt = Scheduler(_opt, schedule, grad_scales)
    fuzzy_loss = FuzzyLoss(KLDivLoss(reduction='batchmean'), len(parser.atom_tokenizer), 0.1,
                           ignore_index=[parser.atom_tokenizer.pad_token_id, parser.atom_tokenizer.sos_token_id])

    if model_path is not None:
        step_num, opt_dict, init_epoch = load_model(parser, model_path)
        opt.step_num = step_num
        opt.lr = opt.schedule(opt.step_num)
        opt.opt.load_state_dict(opt_dict)
    else:
        init_epoch = 0

    if save_to_dir is None:
        save_to_dir = './stored_models'

    for e in range(init_epoch, num_epochs):
        # epoch settings
        validate = True if e == 4 or (e % 20 == 0 and e != 0) else False
        save = True if e == 4 or (e % 10 == 0 and e != 0) else True if e == num_epochs - 1 else False
        epoch_lr = opt.lr
        linking_weight = 0.5

        with open(f'{save_to_dir}/{version}/log.txt', 'a') as stream:
            logprint('=' * 64, [stream])
            logprint(f'Epoch {e}', [stream])
            logprint(' ' * 50 + f'LR: {epoch_lr}', [stream])
            logprint(' ' * 50 + f'LW: {linking_weight}', [stream])
            logprint('-' * 64, [stream])
            supertagging_loss, linking_loss = parser.train_epoch(train_dl, fuzzy_loss, opt, linking_weight)
            logprint(f' Supertagging Loss:\t\t{supertagging_loss:5.2f}', [stream])
            logprint(f' Linking Loss:\t\t\t{linking_loss:5.2f}', [stream])
            if validate:
                with open(f'{save_to_dir}/{version}/val_log.txt', 'a') as valstream:
                    logprint('-' * 64, [stream, valstream])
                    sentence_ac, atom_ac, link_ac = parser.eval_epoch(val_dl, link=linking_weight != 0)
                    logprint(f' Sentence Accuracy:\t\t{(sentence_ac * 100):6.2f}', [stream, valstream])
                    logprint(f' Atom Accuracy:\t\t\t{(atom_ac * 100):6.2f}', [stream, valstream])
                    logprint(f' Link Accuracy:\t\t\t{(link_ac * 100):6.2f}', [stream, valstream])
            logprint('\n', [stream])

            if save:
                print('\tSaving')
                torch.save({'model_state_dict': parser.state_dict(),
                            'opt_state_dict': opt.opt.state_dict(),
                            'step': opt.step_num,
                            'epoch': e},
                           f'{save_to_dir}/{version}/{e}.model')
