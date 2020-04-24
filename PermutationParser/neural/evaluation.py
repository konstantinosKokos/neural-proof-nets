from PermutationParser.neural.model import Parser
from PermutationParser.neural.schedules import DataLoader
from PermutationParser.parsing.utils import (Analysis, sample_to_analysis, get_polarities_and_indices, polish_fn)
from PermutationParser.neural.utils import Tokenizer, AtomTokenizer
from PermutationParser.data.sample import load_stored, Sample
import torch

from typing import List, Callable, Tuple
from functools import reduce

from operator import eq, add
from editdistance import distance


def make_stuff() -> Tuple[Parser, List[Sample]]:
    train, dev, test = load_stored()
    parser = Parser(AtomTokenizer(train+dev+test), Tokenizer(), 768, 128, 'cpu')
    parser.load_state_dict(torch.load('./stored_models/dd/200.model', map_location='cpu')['model_state_dict'])
    return parser, test


def infer_dataset(model: Parser, data: List[Sample], beam_size: int = 5, batch_size: int = 64) -> List[List[Analysis]]:
    ret = []
    start_from = 0
    while start_from < len(data):
        batch = data[start_from: min(start_from + batch_size, len(data))]
        start_from += batch_size
        batch_analyses = model.infer([' '.join(sample.words) for sample in batch], beam_size)
        ret += batch_analyses
    return ret


def data_to_analyses(data: List[Sample]) -> List[Analysis]:
    return [sample_to_analysis(sample) for sample in data]


# Boolean Comparisons
def types_correct(x: Analysis, y: Analysis) -> bool:
    return x.types == y.types and x.conclusion == y.conclusion


def lambdas_correct(x: Analysis, y: Analysis) -> bool:
    return x.lambda_term == y.lambda_term and x.types == y.types and x.conclusion == y.conclusion


def match_in_beam(beam: List[Analysis], correct: Analysis,
                  comparison: Callable[[Analysis, Analysis], bool]) -> bool:
    return any(list(map(lambda x: comparison(x, correct), beam)))


def matches_in_beams(beams: List[List[Analysis]], corrects: List[Analysis],
                     comparison: Callable[[Analysis, Analysis], bool]) -> int:
    return len(list(filter(lambda equal: equal,
                           map(lambda beam, correct: match_in_beam(beam, correct, comparison),
                               beams,
                               corrects))))


def measure_lambda_accuracy(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    return matches_in_beams(beams, corrects, lambdas_correct) / len(beams)


def measure_typing_accuracy(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    return matches_in_beams(beams, corrects, types_correct) / len(beams)


# Float Comparisons
def token_accuracy(x: Analysis, y: Analysis) -> Tuple[int, int]:
    return (len(list(filter(lambda equal: equal,
                            map(lambda pred, true: eq(pred, true),
                                [t.depolarize() for t in x.types + [x.conclusion]],
                                [t.depolarize() for t in y.types + [y.conclusion]])))),
            len(x.types) + 1)


def best_in_beam(beam: List[Analysis], correct: Analysis, comp: Callable[[Analysis, Analysis], Tuple[int, int]])  \
        -> Tuple[int, int]:
    return max(list(map(lambda x: comp(x, correct), beam)))


def measure_token_accuracy(beams: List[List[Analysis]], corrects: List[Analysis]) -> float:
    best, total = list(zip(*list(map(lambda beam, correct:
                                     best_in_beam(beam, correct, token_accuracy),
                                     beams, corrects))))
    return reduce(add, best) / reduce(add, total)
