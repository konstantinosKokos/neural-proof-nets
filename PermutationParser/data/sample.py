import pickle
from dataclasses import dataclass
from typing import List, Set, Tuple
from PermutationParser.parsing.milltypes import WordTypes

ProofNet = Set[Tuple[int, int]]
Matrix = List[List[bool]]
Matrices = List[Matrix]
ints = List[int]
strs = List[str]


@dataclass
class Sample:
    words: strs
    matrices: Matrices
    positive_ids: List[ints]
    negative_ids: List[ints]
    polish: strs
    types: WordTypes
    proof: ProofNet

    def __hash__(self) -> int:
        words = tuple(self.words).__hash__()
        matrices = tuple(map(lambda matrix: tuple(map(tuple, matrix)), self.matrices)).__hash__()
        positive_ids = tuple(map(tuple, self.positive_ids)).__hash__()
        negative_ids = tuple(map(tuple, self.negative_ids)).__hash__()
        polished = tuple(self.polish).__hash__()
        return (words, matrices, positive_ids, negative_ids, polished).__hash__()


def load_stored(file: str = './processed.p'):
    with open(file, 'rb') as f:
        print('Opening pre-processed samples.')
        return pickle.load(f)
