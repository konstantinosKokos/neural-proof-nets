from ..neural.model import Parser
from ..data.preprocessing import Sample, load_stored
from ..parsing.postprocessing import Analysis
from ..neural.utils import make_atom_mapping, AtomTokenizer, Tokenizer

import torch

from typing import Callable
from functools import reduce
from operator import eq, add


def make_stuff() -> tuple[Parser, list[Sample]]:
    train, dev, test = load_stored('./processed.p')
    atom_map = make_atom_mapping(train+dev+test)
    parser = Parser(AtomTokenizer(atom_map), Tokenizer(), device='cuda')
    parser.load_state_dict(torch.load('./stored_models/model_weights.p',
                                      map_location='cuda')['model_state_dict'])
    return parser, sorted(list(filter(lambda x: len(x.polish) < 140, test)), key=lambda x: len(x.polish))


def infer_dataset(model: Parser, data: list[Sample], beam_size: int, batch_size: int) -> list[list[Analysis]]:
    ret = []
    start_from = 0
    while start_from < len(data):
        batch = data[start_from: min(start_from + batch_size, len(data))]
        start_from += batch_size
        if beam_size == 1:
            batch_analyses = model.infer([' '.join(sample.words) for sample in batch], beam_size,
                                         max_decode_length=max([len(s.polish) for s in batch]))
        else:
            batch_analyses = model.infer([' '.join(sample.words) for sample in batch], beam_size)
        ret += batch_analyses
    return ret


def oracle_run(model: Parser, data: list[Sample], batch_size: int) -> list[list[Analysis]]:
    ret = []
    start_from = 0
    while start_from < len(data):
        batch = data[start_from: min(start_from + batch_size, len(data))]
        start_from += batch_size
        batch_analyses = model.parse_with_oracle(batch)
        ret += batch_analyses
    return ret


def data_to_analyses(data: list[Sample]) -> list[Analysis]:
    return [Analysis.from_sample(sample) for sample in data]


def types_correct(x: Analysis, y: Analysis) -> bool:
    return x.types == y.types


def term_correct(x: Analysis, y: Analysis, check_decoration: bool) -> bool:
    y_term = y.to_proofnet().print_term(show_words=False, show_types=False, show_decorations=check_decoration)
    try:
        x_term = x.to_proofnet().print_term(show_words=False, show_types=False, show_decorations=check_decoration)
        return x_term == y_term
    except TypeError:
        return False


def passing(x: Analysis) -> bool:
    return x.traceback is None


def invariance(x: Analysis) -> bool:
    return x.valid()


def match_in_beam(beam: list[Analysis], correct: Analysis, cmp: Callable[[Analysis, Analysis], bool]) -> bool:
    return any((cmp(x, correct) for x in beam))


def matches_in_beams(beams: list[list[Analysis]], corrects: list[Analysis],
                     cmp: Callable[[Analysis, Analysis], bool]) -> int:
    return len(list(filter(lambda equal: equal,
                           (match_in_beam(beam, correct, cmp) for beam, correct in zip(beams, corrects)))))


def measure_lambda_accuracy(beams: list[list[Analysis]], corrects: list[Analysis], check_decorations: bool) -> float:
    if check_decorations:
        comp = lambda x, y: term_correct(x, y, True)
    else:
        comp = lambda x, y: term_correct(x, y, False)
    return matches_in_beams(beams, corrects, comp) / len(beams)


def measure_typing_accuracy(beams: list[list[Analysis]], corrects: list[Analysis]) -> float:
    return matches_in_beams(beams, corrects, types_correct) / len(beams)


def measure_coverage(beams: list[list[Analysis]]) -> float:
    return len(list(filter(lambda beam: any(map(passing, beam)), beams))) / len(beams)


def measure_inv_correct(beams: list[list[Analysis]]) -> float:
    return len(list(filter(lambda beam: any(map(invariance, beam)), beams))) / len(beams)


def token_accuracy(x: Analysis, y: Analysis) -> tuple[int, int]:
    if x.types is None:
        return 0, len(y.types)
    return len(list(filter(lambda pt: eq(*pt), zip(x.types, y.types)))), len(y.types)


def best_in_beam(beam: list[Analysis], correct: Analysis, comp: Callable[[Analysis, Analysis], tuple[int, int]])  \
        -> tuple[int, int]:
    return max(list(map(lambda x: comp(x, correct), beam)))


def measure_token_accuracy(beams: list[list[Analysis]], corrects: list[Analysis]) -> float:
    best, total = list(zip(*list(map(lambda beam, correct:
                                     best_in_beam(beam, correct, token_accuracy),
                                     beams, corrects))))
    return reduce(add, best) / reduce(add, total)


def fill_table(kappas: list[int]) -> None:
    parser, data = make_stuff()

    truths = data_to_analyses(data)
    for k in kappas:
        print('=' * 64)
        print(f'{k=}')
        print('=' * 64)
        predictions = infer_dataset(parser, data, k, 256)
        coverage = measure_coverage(predictions)/len(predictions)
        print(f'{coverage=}')
        invc = measure_inv_correct(predictions)/len(predictions)
        print(f'{invc=}')
        token_acc = measure_token_accuracy(predictions, truths)
        print(f'{token_acc=}')
        typing_acc = measure_typing_accuracy(predictions, truths)
        print(f'{typing_acc=}')
        lambda_acc = measure_lambda_accuracy(predictions, truths, False)
        print(f'{lambda_acc=}')
        lambda_dec_acc = measure_lambda_accuracy(predictions, truths, True)
        print(f'{lambda_dec_acc=}')


def do_oracle_run():
    parser, data = make_stuff()
    return oracle_run(parser, data, 512), data_to_analyses(data)
