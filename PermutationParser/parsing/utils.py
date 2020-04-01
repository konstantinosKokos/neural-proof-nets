from dataclasses import dataclass
from functools import reduce
from typing import Optional, List, Tuple, Dict

from PermutationParser.data.constants import ModDeps
from PermutationParser.data.preprocessing import strs, MWU, add, sep, index_from_polish, polish_fn, Atoms, ints
from PermutationParser.neural.utils import AtomTokenizer, tensorize_batch_indexers, LongTensor
from PermutationParser.parsing.milltypes import (BoxType, DiamondType, WordType, polish_to_type,
                                                 get_polarities_and_indices, polarize_and_index_many,
                                                 polarize_and_index, invariance_check)

WordTypes = List[WordType]
OWordType = Optional[WordType]
OWordTypes = List[OWordType]


@dataclass(init=False)
class Analysis:
    words: Optional[strs]
    types: Optional[WordTypes]
    conclusion: Optional[WordType]
    polishes: Optional[strs]
    atom_set: Optional[Atoms]
    positive_ids: Optional[List[ints]]
    negative_ids: Optional[List[ints]]
    idx_to_polish: Optional[Dict[int, int]]
    proof: Optional[Dict[int, int]]

    def __init__(self, words=None, types=None, conclusion=None, polishes=None, atom_set=None,
                 positive_ids=None, negative_ids=None, idx_to_polish=None, proof=None):
        self.words = words
        self.types = types
        self.conclusion = conclusion
        self.polishes = polishes
        self.atom_set = atom_set
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids
        self.idx_to_polish = idx_to_polish
        self.proof = proof

    def __len__(self):
        return len(self.polishes) if self.polishes is not None else 0

    def get_ids(self) -> Tuple[List[List[int]], List[List[int]]]:
        if self.positive_ids is None:
            return [], []
        return self.positive_ids, self.negative_ids

    def fill_matches(self, matrices: List[ints]):
        pos: ints
        neg: ints
        match: ints
        pnet: Dict[int, int] = dict()

        if self.positive_ids is None or self.negative_ids is None:
            return None

        for pos, neg, match in zip(self.positive_ids, self.negative_ids, matrices):
            for i, p in enumerate(pos):
                n_idx = match[i]
                n = neg[n_idx]
                pnet[self.idx_to_polish[p]] = self.idx_to_polish[n]
        self.proof = pnet


class TypeParser(object):
    def __init__(self, atom_tokenizer: AtomTokenizer):
        self.operators = {k for k in atom_tokenizer.atom_map.keys() if k.lower() == k and k != '_'}
        self.operator_classes = {k: BoxType if k in ModDeps else DiamondType for k in self.operators if k != '→'}

    def make_partial_analysis(self, sentence: str, polishes: List[strs]) -> Analysis:
        analysis = Analysis(words=sentence.split())
        types = self.sent_to_types(polishes)
        if types is not None:
            analysis.types = types[1:]
            analysis.conclusion = types[0]
        polarized = self.polarize_sent(types)
        atoms_and_indices = self.get_atomset_and_indices(polarized)
        if atoms_and_indices is not None:
            polished, atom_set, positives, negatives, polish_from_idx = atoms_and_indices
            analysis.polishes = polished
            analysis.atom_set = atom_set
            analysis.positive_ids = positives
            analysis.negative_ids = negatives
            analysis.idx_to_polish = polish_from_idx
        return analysis

    def make_batch_partial_analyses(self, sents: List[str], polishes: List[List[strs]]) -> List[Analysis]:
        return [self.make_partial_analysis(s, p) for s, p in zip(sents, polishes)]

    def analyses_to_indices(self, analyses: List[Analysis]) -> Tuple[List[List[LongTensor]], List[List[LongTensor]]]:
        positive_ids, negative_ids = list(zip(*[analysis.get_ids() for analysis in analyses]))
        return tensorize_batch_indexers(positive_ids), tensorize_batch_indexers(negative_ids)

    def polish_to_type(self, polished: strs) -> Optional[WordType]:
        try:
            return polish_to_type(polished, self.operators, self.operator_classes)
        except Exception:
            return None

    def sent_to_types(self, sent: Optional[List[strs]]) -> Optional[OWordTypes]:
        return None if sent is None else [self.polish_to_type(subseq) for subseq in sent]

    @staticmethod
    def polarize_sent(sent: Optional[OWordTypes]) -> Optional[WordTypes]:
        if sent is None or any(map(lambda wordtype: wordtype is None, sent)):
            return None
        idx, wordtypes = polarize_and_index_many(sent[1:], index=0)
        _, conclusion = polarize_and_index(sent[0], polarity=False, index=idx)
        if invariance_check(wordtypes, conclusion):
            return [conclusion] + wordtypes
        return None

    @staticmethod
    def get_atomset_and_indices(sent: Optional[WordTypes]) \
            -> Optional[Tuple[strs, Atoms, List[ints], List[ints], Dict[int, int]]]:
        if sent is None:
            return None
        atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, sent[1:])))))
        negative, positive = list(map(lambda x: reduce(add, x), atoms))
        negative += get_polarities_and_indices(sent[0])[1]

        local_atom_set = list(set(map(lambda x: x[0], positive + negative)))
        positive_sep = sep(positive, local_atom_set)
        negative_sep = sep(negative, local_atom_set)

        polished = polish_fn(sent)
        positional_ids = index_from_polish(polished, offset=-1)
        polish_from_index = {v: k for k, v in positional_ids.items()}

        positive_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                positive_sep))
        negative_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                negative_sep))

        return polished, local_atom_set, positive_ids, negative_ids, polish_from_index
