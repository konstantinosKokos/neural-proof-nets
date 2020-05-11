from PermutationParser.neural.model import Parser
from PermutationParser.parsing.utils import Analysis, sample_to_analysis
from PermutationParser.neural.utils import Tokenizer, AtomTokenizer
from PermutationParser.data.sample import load_stored, Sample
import torch

from typing import List, Callable, Tuple, Optional
from functools import reduce

from operator import eq, add


def make_stuff() -> Tuple[Parser, List[Sample]]:
    train, dev, test = load_stored('./processed_old.p')
    parser = Parser(AtomTokenizer(train+dev+test), Tokenizer(), 768, 768, 'cuda')
    parser.load_state_dict(torch.load('./stored_models/5-3-12-768-32-nll/140.model',
                                      map_location='cuda')['model_state_dict'])
    return parser, sorted(list(filter(lambda x: len(x.polish) < 100, test)), key=lambda x: len(x.polish))


def infer_dataset(model: Parser, data: List[Sample], beam_size: int, batch_size: int) -> List[List[Analysis]]:
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


def lambdas_correct(x: Analysis, y: Analysis, check_decoration: bool) -> bool:
    if check_decoration:
        return x.lambda_term == y.lambda_term and x.types == y.types and x.conclusion == y.conclusion
    else:
        return x.lambda_term_no_dec == y.lambda_term_no_dec


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


# Float Comparisons
def token_accuracy(x: Optional[Analysis], y: Analysis) -> Tuple[int, int]:
    if x is None:
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


def fill_table(parser: Parser, kappas: List[int], data: List[Sample]):
    truth = data_to_analyses(data)
    for k in kappas:
        print(f'{k =}')
        predictions = infer_dataset(parser, data, k, 256)
        token_acc = measure_token_accuracy(predictions, truth)
        print(f'{token_acc =}')
        typing_acc = measure_typing_accuracy(predictions, truth)
        print(f'{typing_acc =}')
        lambda_acc = measure_lambda_accuracy(predictions, truth, False)
        print(f'{lambda_acc =}')
        lambda_dec_acc = measure_lambda_accuracy(predictions, truth, True)
        print(f'{lambda_dec_acc =}')

