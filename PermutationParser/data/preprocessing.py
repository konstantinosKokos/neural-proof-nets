import sys
from functools import reduce
from itertools import chain
from operator import add
from typing import Dict, Optional

from PermutationParser.data.constants import PtDict, CatDict
from PermutationParser.data.sample import *
from PermutationParser.parsing import milltypes
from PermutationParser.parsing.milltypes import (WordType, AtomicType, get_polarities_and_indices, polish,
                                                 PolarizedType)

sys.modules['LassyExtraction.milltypes'] = milltypes

Atoms = List[AtomicType]
MWU = AtomicType('_MWU')


def make_atom_set() -> Atoms:
    return list(PtDict.values()) + list(CatDict.values()) + [MWU]


def polish_fn(types: List[WordType], sos_symbol: str = '[SOS]', sep_symbol: str = '[SEP]') -> strs:
    return [sos_symbol] + f' {sep_symbol} '.join(map(polish, types)).split() + [sep_symbol]


def preprocess_pairs(words: strs, types: List[WordType]) -> Tuple[strs, List[WordType]]:
    words = [word.split() for word in words]
    types = [[wordtype] + [MWU for _ in range(len(words[i]) - 1)] for i, wordtype in enumerate(types)]
    return list(chain.from_iterable(words)), list(chain.from_iterable(types))


def sep(_atoms: List[Tuple[AtomicType, int]], atom_set: Atoms) -> List[List[Tuple[AtomicType, int]]]:
    return [list(filter(lambda p: p[0] == a, _atoms)) for a in atom_set]


def preprocess(words: strs, types: List[WordType], proof: ProofNet, atom_set: Optional[Atoms] = None) -> Sample:
    def get_conclusion(_atoms: List[Tuple[AtomicType, int]], _proof: ProofNet) -> Tuple[AtomicType, int]:
        antecedents = set(map(lambda x: x[1], _atoms))
        conclusion_id = list(set(map(lambda x: x[1],
                                     _proof)).
                             union(set(map(lambda x: x[0],
                                           _proof))).
                             difference(antecedents))[0]
        conclusion_pair = list(filter(lambda pair: pair[1] == conclusion_id, _proof))[0][0]
        conclusion_atom = list(filter(lambda a: a[1] == conclusion_pair, _atoms))[0][0]
        return conclusion_atom, conclusion_id

    if len(types) == 1:
        return None

    if atom_set is None:
        atom_set = make_atom_set()

    words, types = preprocess_pairs(words, types)

    atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, types)))))
    negative, positive = list(map(lambda x: reduce(add, x), atoms))

    conclusion = get_conclusion(positive + negative, proof)
    negative += [conclusion]

    positive_sep = sep(positive, atom_set)
    negative_sep = sep(negative, atom_set)

    matrices = list(map(lambda x: convert_matches_to_matrix(x, proof),
                        list(zip(
                            list(map(lambda _sep: [item[1] for item in _sep], positive_sep)),
                            list(map(lambda _sep: [item[1] for item in _sep], negative_sep))))))

    polished = polish_fn([PolarizedType(wordtype=str(conclusion[0]), polarity=False, index=conclusion[1])] + types)
    positional_ids = index_from_polish(polished, offset=-1)

    positive_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                            positive_sep))
    negative_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                            negative_sep))

    return Sample(words=words, matrices=matrices, positive_ids=positive_ids, negative_ids=negative_ids,
                  polish=remove_polarities(polished),
                  types=list(map(str, types)))


def convert_matches_to_matrix(matches: Tuple[ints, ints], proof: ProofNet) -> Matrix:
    def is_match(_i: int, _j: int) -> bool:
        return (_i, _j) in proof

    return [[is_match(i, j) for i in matches[0]] for j in matches[1]]


def index_from_polish(polished: List[str], offset: int) -> Dict[int, int]:
    def is_atom(x: str) -> bool:
        return '(' in x

    def get_idx(x: str) -> int:
        return int(x.split(',')[1].split(')')[0])

    return {get_idx(atom): i + offset for i, atom in enumerate(polished) if is_atom(atom)}


def remove_polarity(indexed: str) -> str:
    return indexed if '(' not in indexed else indexed.split('(')[0]


def remove_polarities(indexed: strs) -> strs:
    return list(map(remove_polarity, indexed))


def main() -> List[Sample]:
    with open('./maximal.p', 'rb') as f:
        words, types, proofs = pickle.load(f)

    samples = list(filter(lambda x: x is not None,
                          map(lambda w, t, p: preprocess(w, t, p), words, types, proofs)))
    with open('./processed.p', 'wb') as f:
        pickle.dump(samples, f)
        print('Saved pre-processed samples.')
        return samples
