from PermutationParser.neural.model import Parser
from PermutationParser.parsing.utils import Analysis, sample_to_analysis
from PermutationParser.neural.utils import Tokenizer, AtomTokenizer
from PermutationParser.data.sample import load_stored, Sample
import torch

from typing import List, Callable, Tuple, Optional
from functools import reduce

from operator import eq, add


def make_stuff() -> Tuple[Parser, List[Sample]]:
    train, dev, test = load_stored('./processed.p')
    parser = Parser(AtomTokenizer(train+dev+test), Tokenizer(), 768, 256, 'cuda')
    parser.load_state_dict(torch.load('./stored_models/3-1-8-256-32-nll/280.model',
                                      map_location='cuda')['model_state_dict'])
    return parser, sorted(list(filter(lambda x: len(x.polish) < 100, test)), key=lambda x: len(x.polish))


def infer_dataset(model: Parser, data: List[Sample], beam_size: int, batch_size: int) -> List[List[Analysis]]:
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


def data_to_analyses(data: List[Sample]) -> List[Analysis]:
    return [sample_to_analysis(sample) for sample in data]


# Boolean Comparisons
def types_correct(x: Analysis, y: Analysis) -> bool:
    return x.types == y.types and x.conclusion == y.conclusion


def lambdas_correct(x: Analysis, y: Analysis, check_decoration: bool) -> bool:
    if check_decoration:
        return x.lambda_term == y.lambda_term and x.types == y.types and x.conclusion == y.conclusion
    else:
        return x.lambda_term_no_dec == y.lambda_term_no_dec


def not_failed(x: Analysis, y: Analysis) -> bool:
    return x.types is not None and x.conclusion is not None


def invariance(x: Analysis, y: Analysis) -> bool:
    return x.positive_ids is not None and x.negative_ids is not None


def match_in_beam(beam: List[Analysis], correct: Analysis,
                  comparison: Callable[[Analysis, Analysis], bool]) -> bool:
    return any(list(map(lambda x: comparison(x, correct), beam)))


def matches_in_beams(beams: List[List[Analysis]], corrects: List[Analysis],
                     comparison: Callable[[Analysis, Analysis], bool]) -> int:
    return len(list(filter(lambda equal: equal,
                           map(lambda beam, correct: match_in_beam(beam, correct, comparison),
                               beams,
                               corrects))))


def measure_lambda_accuracy(beams: List[List[Analysis]], corrects: List[Analysis], check_decorations: bool) -> float:
    if check_decorations:
        comp = lambda x, y: lambdas_correct(x, y, True)
    else:
        comp = lambda x, y: lambdas_correct(x, y, False)
    return matches_in_beams(beams, corrects, comp) / len(beams)


def measure_typing_accuracy(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    return matches_in_beams(beams, corrects, types_correct) / len(beams)


def measure_non_failed(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    return matches_in_beams(beams, corrects, not_failed)


def measure_inv_correct(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    return matches_in_beams(beams, corrects, invariance)


# Float Comparisons
def token_accuracy(x: Analysis, y: Analysis) -> Tuple[int, int]:
    if x.types is None or x.conclusion is None:
        return 0, len(y.types) + 1
    return (len(list(filter(lambda equal: equal,
                            map(lambda pred, true: eq(pred, true),
                                [t.depolarize() for t in x.types + [x.conclusion]],
                                [t.depolarize() for t in y.types + [y.conclusion]])))),
            len(x.types) + 1)


def best_in_beam(beam: List[Analysis], correct: Analysis, comp: Callable[[Analysis, Analysis], Tuple[int, int]])  \
        -> Tuple[int, int]:
    if not len(beam):
        return token_accuracy(None, correct)
    return max(list(map(lambda x: comp(x, correct), beam)))


def measure_token_accuracy(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    best, total = list(zip(*list(map(lambda beam, correct:
                                     best_in_beam(beam, correct, token_accuracy),
                                     beams, corrects))))
    return reduce(add, best) / reduce(add, total)


def fill_table(kappas: List[int], lens: List[int]):
    parser, data = make_stuff()

    truths = data_to_analyses(data)
    for k in kappas:
        print(f'{k =}')
        predictions = infer_dataset(parser, data, k, 256)
        ok = measure_non_failed(predictions, truths)/len(predictions)
        print(f'{ok =}')
        invc = measure_inv_correct(predictions, truths)/len(predictions)
        print(f'{invc=}')
        token_acc = measure_token_accuracy(predictions, truths)
        print(f'{token_acc =}')
        typing_acc = measure_typing_accuracy(predictions, truths)
        print(f'{typing_acc =}')
        lambda_acc = measure_lambda_accuracy(predictions, truths, False)
        print(f'{lambda_acc =}')
        lambda_dec_acc = measure_lambda_accuracy(predictions, truths, True)
        print(f'{lambda_dec_acc =}')
        for i in range(len(lens) + 1):
            maxlen = lens[i] if i != len(lens) else 1000
            minlen = 0 if i == 0 else lens[i-1]
            print(f'{minlen =}')
            print(f'{maxlen =}')
            tmp = [(truth, preds) for truth, preds in zip(truths, predictions) if minlen < len(truth.words) <= maxlen]
            truths_l, preds_l = list(zip(*tmp))
            ok_l = measure_non_failed(preds_l, truths_l)
            print(f'{ok_l =}')
            token_acc_l = measure_token_accuracy(preds_l, truths_l)
            print(f'{token_acc_l =}')
            typing_acc_l = measure_typing_accuracy(preds_l, truths_l)
            print(f'{typing_acc_l =}')
            lambda_acc_l = measure_lambda_accuracy(preds_l, truths_l, False)
            print(f'{lambda_acc_l =}')
            lambda_dec_acc_l = measure_lambda_accuracy(preds_l, truths_l, True)
            print(f'{lambda_dec_acc_l =}')
    return predictions, truths


