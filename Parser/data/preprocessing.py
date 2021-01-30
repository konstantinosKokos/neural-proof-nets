"""
Contains utility functions to convert the aethel dataset to inputs for the neural proof net.
"""

from dataclasses import dataclass

from itertools import chain
from functools import reduce
from operator import add

from LassyExtraction.aethel import ProofNet, AxiomLinks
from LassyExtraction.milltypes import (WordType, AtomicType, binarize_polish, FunctorType, get_polarities_and_indices,
                                       ModalType)
from LassyExtraction.extraction import CatDict, PtDict
from typing import Optional

MWU = AtomicType('_MWU')

_atom_collations = {'SPEC': 'NP'}


def make_atom_set() -> list[AtomicType]:
    pts = set(PtDict.values())
    cats = set(CatDict.values())
    rem = set(map(AtomicType, _atom_collations.keys()))
    return sorted(pts.union(cats).difference(rem).union({MWU}), key=lambda a: str(a))


_atom_set = make_atom_set()


def polish(wordtype: WordType, sep: str) -> list[str]:
    return binarize_polish(wordtype.polish()) + [sep]


def polish_seq(types: list[WordType], sos: str = '[SOS]', sep: str = '[SEP]') -> list[str]:
    return [sos] + sum([polish(t, sep) for t in types], [])


def pad_mwus(words: list[str], types: list[WordType]) -> tuple[list[str], list[WordType]]:
    words = [w.split() for w in words]
    types = [[wt] + [MWU] * (len(w)-1) for w, wt in zip(words, types)]
    return list(chain.from_iterable(words)), list(chain.from_iterable(types))


def separate(atoms: list[tuple[AtomicType, int]], atom_set: list[AtomicType]) -> list[list[tuple[AtomicType, int]]]:
    return [list(filter(lambda p: p[0] == a, atoms)) for a in atom_set]


def convert_matches_to_matrix(matches: tuple[list[int], list[int]], links: AxiomLinks) -> list[list[bool]]:
    def is_match(_i: int, _j: int) -> bool:
        return (_i, _j) in links
    return [[is_match(i, j) for j in matches[1]] for i in matches[0]]


def is_atom(x: str) -> bool:
    return '(' in x


def idx_from_polish(polished: list[str], offset: int) -> dict[int, int]:
    def get_idx(x: str) -> int:
        return int(x.split(',')[1].split(')')[0])

    return {get_idx(atom): i + offset for i, atom in enumerate(polished) if is_atom(atom)}


def remove_polarity(indexed: str) -> str:
    return indexed if not is_atom(indexed) else indexed.split('(')[0]


def remove_polarities(indexed: list[str]) -> list[str]:
    return [remove_polarity(i) for i in indexed]


def collate_atom(atom: str) -> str:
    return _atom_collations[atom] if atom in _atom_collations.keys() else atom


def collate_type(wordtype: WordType) -> WordType:
    if isinstance(wordtype, AtomicType):
        wordtype.type = collate_atom(wordtype.type)
        return wordtype
    elif isinstance(wordtype, FunctorType):
        collate_type(wordtype.argument)
        collate_type(wordtype.result)
        return wordtype
    elif isinstance(wordtype, ModalType):
        collate_type(wordtype.content)
        return wordtype
    else:
        raise TypeError(f'Unexpected argument {wordtype} of type {type(wordtype)}')


@dataclass
class Sample:
    words: list[str]
    types: list[WordType]
    matrices: list[list[list[bool]]]
    pos_ids: list[list[int]]
    neg_ids: list[list[int]]
    polish: list[str]
    proof: AxiomLinks
    name: Optional[str]

    @staticmethod
    def from_proofnet(pn: ProofNet) -> 'Sample':

        words = pn.proof_frame.get_words()
        types = list(map(collate_type, pn.proof_frame.get_types()))
        words, types = pad_mwus(words, types)

        atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, types)))))
        negative, positive = list(map(lambda x: reduce(add, x), atoms))
        conclusion = pn.proof_frame.conclusion
        negative += [(collate_type(conclusion), conclusion.index)]

        p_sep = separate(positive, _atom_set)
        n_sep = separate(negative, _atom_set)

        matrices = list(map(lambda x: convert_matches_to_matrix(x, pn.axiom_links),
                            list(zip(
                                list(map(lambda _sep: [item[1] for item in _sep], p_sep)),
                                list(map(lambda _sep: [item[1] for item in _sep], n_sep))))))

        polished = polish_seq([conclusion] + types)
        p_to_ids = idx_from_polish(polished, 0)
        pos_ids = list(map(lambda idxs: list(map(lambda atom: p_to_ids[atom[1]], idxs)), p_sep))
        neg_ids = list(map(lambda idxs: list(map(lambda atom: p_to_ids[atom[1]], idxs)), n_sep))
        return Sample(words, types, matrices, pos_ids, neg_ids, remove_polarities(polished), pn.axiom_links, pn.name)

    @staticmethod
    def from_dataset(dataset: list[list[ProofNet]]) -> tuple[list[list['Sample']], list[str]]:
        samples = [[Sample.from_proofnet(pn) for pn in pns] for pns in dataset]
        atoms = set([t for sample in sum(samples, []) for t in sample.polish])
        return samples, ['[PAD]'] + sorted(atoms)


def load_stored(file: str = './processed.p') -> tuple[list[Sample], list[Sample], list[Sample]]:
    import pickle
    with open(file, 'rb') as f:
        print('Opening pre-processed samples.')
        return pickle.load(f)
