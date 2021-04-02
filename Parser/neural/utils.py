from itertools import chain, groupby
from typing import Mapping, Iterable, Optional

import torch
from torch import Tensor, LongTensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence as _pad_sequence
from torch.nn.functional import nll_loss, cross_entropy
from transformers import BertTokenizer

from ..data.preprocessing import Sample


class Tokenizer:
    def __init__(self):
        self.core = BertTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')

    def encode_sample(self, sample: Sample) -> list[int]:
        return self.core.encode(' '.join(sample.words))

    def encode_samples(self, samples: list[Sample]) -> list[list[int]]:
        return [self.encode_sample(sample) for sample in samples]

    def encode_sent(self, sent: str) -> list[int]:
        return self.core.encode(sent)

    def encode_sents(self, sents: list[str]) -> list[list[int]]:
        return [self.encode_sent(sent) for sent in sents]


def pad_sequence(x, **kwargs):
    return _pad_sequence(x, batch_first=True, **kwargs)


def make_sinkhorn_inputs(bsd_tensor: Tensor, positional_ids: list[list[Tensor]], device: str = 'cpu') \
        -> list[list[Tensor]]:
    """

    :param bsd_tensor:
        Tensor of shape batch size \times sequence length \times feature dimensionality.
    :param positional_ids:
        A List of batch_size elements, each being a List of num_atoms LongTensors.
        Each LongTensor in positional_ids[b][a] indexes the location of atoms of type a in sentence b.
    :param device:
    :return:
    """

    return [[bsd_tensor.select(0, index=i).index_select(0, index=atom.to(device)) for atom in sentence]
            for i, sentence in enumerate(positional_ids)]


def tensorize_sentence_indexers(indexers: list[list[int]]) -> list[Tensor]:
    return [LongTensor(index) for index in indexers]


def tensorize_batch_indexers(sentences: list[list[list[int]]]) -> list[list[Tensor]]:
    return [tensorize_sentence_indexers(sentence) for sentence in sentences]


def sents_to_batch(sents: list[str], tokenizer: Tokenizer) -> Tensor:
    tokens = tokenizer.encode_sents(sents)
    return pad_sequence([LongTensor(token_seq) for token_seq in tokens], padding_value=tokenizer.core.pad_token_id)


class AtomTokenizer(object):
    def __init__(self, atom_map: Optional[Mapping[str, int]] = None):
        self.atom_map = read_atom_mapping() if atom_map is None else atom_map
        self.inverse_atom_map = {v: k for k, v in self.atom_map.items()}
        self.sep_token = '[SEP]'
        self.sos_token = '[SOS]'
        self.pad_token = '[PAD]'
        self.sep_token_id = self.atom_map[self.sep_token]
        self.sos_token_id = self.atom_map[self.sos_token]
        self.pad_token_id = self.atom_map[self.pad_token]

    def __len__(self) -> int:
        return len(self.atom_map)

    def convert_atoms_to_ids(self, atoms: Iterable[str]) -> list[int]:
        return [self.atom_map[str(atom)] for atom in atoms]

    def convert_sents_to_ids(self, sentences: Iterable[Iterable[str]]) -> list[list[int]]:
        return [self.convert_atoms_to_ids(atoms) for atoms in sentences]

    def convert_ids_to_atoms(self, ids: Iterable[int]) -> list[str]:
        return [self.inverse_atom_map[int(i)] for i in ids]

    def convert_batch_ids_to_atoms(self, batch_ids: Iterable[Iterable[int]],
                                   max_lens: Optional[list[int]] = None) -> list[Optional[list[str]]]:
        if max_lens is not None:
            return [self.drop_padded_atoms(self.convert_ids_to_atoms(ids), length)
                    for ids, length in zip(batch_ids, max_lens)]
        return [self.convert_ids_to_atoms(ids) for ids in batch_ids]

    def convert_batch_ids_to_polish(self, batch_ids: list[list[int]],
                                    max_lens: list[int]) -> list[Optional[list[list[str]]]]:
        sents = [self.drop_padded_atoms(self.convert_ids_to_atoms(ids), length)
                 for ids, length in zip(batch_ids, max_lens)]
        return [None if sent is None else [t.split() for t in sent] for sent in sents]

    def convert_beam_ids_to_polish(self, beam_ids: list[list[list[int]]], max_lens: list[int]) \
            -> list[list[Optional[list[list[str]]]]]:
        return [self.convert_batch_ids_to_polish(sent, [length] * len(sent))
                for sent, length in zip(beam_ids, max_lens)]

    @staticmethod
    def drop_padded_atoms(atoms: Iterable[str], max_len: int, sep: str = '[SEP]') -> Optional[list[str]]:
        if len(list(filter(lambda x: x == sep, atoms))) < max_len:
            return None
        ret = ' ' + ' '.join(atoms[1:])
        ret = ret.split(f'{sep}')[:max_len]
        return [r[1:-1] for r in ret]

    def samples_to_batch(self, samples: list[Sample], tokenizer: Tokenizer) \
            -> tuple[Tensor, Tensor, list[list[Tensor]], list[list[Tensor]]]:
        return samples_to_batch(samples, self, tokenizer)


Item = tuple[Tensor, Tensor, list[Tensor], list[Tensor], list[list[list[bool]]]]
Batch = tuple[Tensor, Tensor, list[list[Tensor]], list[list[Tensor]], list[Tensor]]


def vectorize_sample(sample: Sample, atokenizer: 'AtomTokenizer', tokenizer: Tokenizer) -> Item:
    symbols = tokenizer.encode_sample(sample)
    embedding_ids = LongTensor(symbols)
    polishes = sample.polish
    _atom_ids = atokenizer.convert_atoms_to_ids(polishes)
    atom_ids = LongTensor(_atom_ids)
    _positives = sample.pos_ids
    _negatives = sample.neg_ids
    positives = tensorize_sentence_indexers(_positives)
    negatives = tensorize_sentence_indexers(_negatives)
    matrices = sample.matrices
    return embedding_ids, atom_ids, positives, negatives, matrices


def batchify_vectorized_samples(inps: list[Item], padding_value_word: int, padding_value_symbol: int,
                                max_difficulty: int, exclude_singular: bool) -> Batch:

    _embedding_ids, _atom_ids, _positives, _negatives, _matrices = list(zip(*inps))
    embedding_ids = pad_sequence(_embedding_ids, padding_value=padding_value_word)
    atom_ids = pad_sequence(_atom_ids, padding_value=padding_value_symbol)
    return embedding_ids, atom_ids, _positives, _negatives, make_permutors(_matrices, max_difficulty, exclude_singular)


def samples_to_batch(samples: list[Sample], atokenizer: 'AtomTokenizer', tokenizer: Tokenizer) \
        -> tuple[Tensor, Tensor, list[list[Tensor]], list[list[Tensor]]]:
    symbols: list[list[int]] = tokenizer.encode_samples(samples)
    embedding_ids: Tensor = pad_sequence([LongTensor(sample) for sample in symbols],
                                         padding_value=tokenizer.core.pad_token_id)

    polishes: list[list[str]] = [sample.polish for sample in samples]
    _atom_ids: list[list[int]] = atokenizer.convert_sents_to_ids(polishes)
    atom_ids: Tensor = pad_sequence([LongTensor(sample) for sample in _atom_ids])

    _positives: list[list[list[int]]] = [sample.pos_ids for sample in samples]
    _negatives: list[list[list[int]]] = [sample.neg_ids for sample in samples]

    positives = tensorize_batch_indexers(_positives)
    negatives = tensorize_batch_indexers(_negatives)
    return embedding_ids, atom_ids, positives, negatives


def make_atom_mapping(samples: list[Sample]) -> Mapping[str, int]:
    polishes: list[str] = list(chain.from_iterable(map(lambda sample: sample.polish, samples)))
    return {**{'[PAD]': 0}, **{p: i + 1 for i, p in enumerate(sorted(set(polishes)))}}


def read_atom_mapping(path: str = './Parser/data/atom_map.txt'):
    atom_map = dict()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            atom, idx = line.split(':')
            atom_map[atom] = int(idx)
    return atom_map


def measure_linking_accuracy(pred: list[Tensor], truth: list[Tensor]) -> tuple[int, int]:
    def measure_one(pred_one: Tensor, truth_one: Tensor) -> tuple[int, int]:
        b, n = pred_one.shape[0:2]
        return torch.sum(pred_one.argmax(dim=-1).eq(truth_one)).item(), b * n

    measurements = list(map(measure_one, pred, truth))
    correct, total = list(zip(*measurements))

    return sum(correct), sum(total)


def measure_supertagging_accuracy(pred: Tensor, truth: Tensor, ignore_idx: int = 0) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    correct_words = torch.ones(pred.size())
    correct_words[pred != truth] = 0
    correct_words[truth == ignore_idx] = 1

    correct_sentences = correct_words.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_words = correct_words.sum().item()
    num_masked_words = len(truth[truth == ignore_idx])

    return (num_correct_sentences, pred.shape[0]), \
           (num_correct_words - num_masked_words, pred.shape[0] * pred.shape[1] - num_masked_words)


def make_permutors(matrices: list[list[list[list[bool]]]], max_difficulty: int, exclude_singular: bool = True) \
        -> list[Tensor]:
    def tensorize_matrix(matrix: list[list[bool]]) -> Tensor:
        return LongTensor([row.index(True) for row in matrix])

    permutors = groupby(sorted(chain.from_iterable(matrices), key=len), key=len)
    return [torch.stack([tensorize_matrix(m) for m in ms]) for s, ms in permutors
            if int(exclude_singular) < s < max_difficulty]


def count_sep(xs: Tensor, sep_id: int, dim: int = -1) -> Tensor:
    return xs.eq(sep_id).sum(dim=dim)


class FuzzyLoss(Module):
    def __init__(self, loss_fn: Module, num_classes: int,
                 mass_redistribution: float, ignore_index: list[int]) -> None:
        super(FuzzyLoss, self).__init__()
        self.loss_fn = loss_fn
        self.nc = num_classes
        self.mass_redistribution = mass_redistribution
        self.ignore_index = ignore_index

    def __call__(self, x: Tensor, y: LongTensor) -> Tensor:
        y = y.flatten(0, -1)
        y_float = torch.zeros(x.shape[0] * x.shape[1], self.nc, device=x.device, dtype=torch.float)
        y_float.fill_(self.mass_redistribution / (self.nc-(1 + len(self.ignore_index))))
        y_float.scatter_(1, y.unsqueeze(1), 1 - self.mass_redistribution)
        mask = torch.zeros_like(y, dtype=torch.bool)
        for idx in self.ignore_index:
            mask = torch.bitwise_or(mask, y == idx)
        y_float[mask.unsqueeze(1).repeat(1, self.nc)] = 0
        return self.loss_fn(torch.log_softmax(x.view(-1, self.nc), dim=-1), y_float)


class NormCrossEntropy(Module):
    def __init__(self, ignore_index: int, sep_id: int):
        super(NormCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.sep_id = sep_id

    def forward(self, predictions: Tensor, truths: Tensor):
        return cross_entropy(predictions.flatten(0, -2), truths.flatten(),
                             reduction='sum', ignore_index=self.ignore_index) / count_sep(truths.flatten(), self.sep_id)


class SinkhornLoss(Module):
    def __init__(self):
        super(SinkhornLoss, self).__init__()

    def forward(self, predictions: list[Tensor], truths: list[Tensor]):
        return sum(nll_loss(link.flatten(0, 1), perm.flatten(), reduction='mean')
                   for link, perm in zip(predictions, truths))


def count_parameters(model: Module) -> int:
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = torch.prod(torch.tensor(param.size()))
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param
