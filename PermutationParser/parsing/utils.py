from dataclasses import dataclass
from functools import reduce
from typing import Optional, List, Tuple

from PermutationParser.data.constants import ModDeps
from PermutationParser.data.preprocessing import (strs, MWU, add, sep, index_from_polish, polish_fn, Atoms, ints,
                                                  Sample, make_atom_set, get_conclusion)
from PermutationParser.neural.utils import AtomTokenizer, tensorize_batch_indexers, LongTensor
from PermutationParser.parsing.milltypes import (BoxType, DiamondType, WordType, polish_to_type,
                                                 get_polarities_and_indices, polarize_and_index_many,
                                                 polarize_and_index, invariance_check)
from PermutationParser.parsing.lambdas import Graph, make_graph, IntMapping, traverse

WordTypes = List[WordType]
OWordType = Optional[WordType]
OWordTypes = List[OWordType]

_atom_set = make_atom_set()


@dataclass(init=False)
class Analysis:
    words: strs
    types: Optional[WordTypes] = None
    conclusion: Optional[WordType] = None
    polish: Optional[strs] = None
    atom_set: Optional[Atoms] = None
    positive_ids: Optional[List[ints]] = None
    negative_ids: Optional[List[ints]] = None
    idx_to_polish: Optional[IntMapping] = None
    axiom_links: Optional[IntMapping] = None
    proof_structure: Optional[Graph] = None
    lambda_term: Optional[str] = None

    def __init__(self, words: strs, types: Optional[WordTypes], conclusion: Optional[WordType], polish: Optional[strs],
                 atom_set: Optional[Atoms], positive_ids: Optional[List[ints]], negative_ids: Optional[List[ints]],
                 idx_to_polish: Optional[IntMapping], axiom_links: Optional[IntMapping] = None,
                 proof_structure: Optional[Graph] = None, lambda_term: str = None):
        self.words = words
        self.types = types
        self.conclusion = conclusion
        self.polish = polish
        self.atom_set = atom_set
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids
        self.idx_to_polish = idx_to_polish
        self.axiom_links = axiom_links
        self.proof_structure = proof_structure
        self.lambda_term = lambda_term

    def __len__(self):
        return len(self.polish) if self.polish is not None else 0

    def __repr__(self):
        return ', '.join([f'{w}: {t}' for w, t in zip(self.words, self.types)]) + f' ⊢ {self.conclusion}'

    def __eq__(self, other: 'Analysis') -> Optional[bool]:
        if any(map(lambda x: x is None, [self.words, self.types, self.axiom_links, other.words, other.types,
                                         other.axiom_links])):
            return None
        return self.words == other.words and self.types == other.types and self.axiom_links == other.axiom_links

    def get_ids(self) -> Tuple[List[List[int]], List[List[int]]]:
        if self.positive_ids is None:
            return [], []
        return self.positive_ids, self.negative_ids

    def fill_matches(self, matrices: List[ints]):
        pos: ints
        neg: ints
        match: ints
        pnet: IntMapping = dict()
        self.proof_structure = make_graph(self.words + ['conc'], self.types, self.conclusion)

        if self.positive_ids is None or self.negative_ids is None:
            return None

        for pos, neg, match in zip(self.positive_ids, self.negative_ids, matrices):
            for i, p in enumerate(pos):
                n_idx = match[i]
                n = neg[n_idx]
                pnet[self.idx_to_polish[p]] = self.idx_to_polish[n]
        if not len(set(pnet.keys())) == len(set(pnet.values())):
            return None

        self.axiom_links = pnet
        self.lambda_term = traverse(self.proof_structure, str(self.conclusion.index),
                                    {str(k): str(v) for k, v in self.axiom_links.items()},
                                    {str(v): str(k) for k, v in self.axiom_links.items()},
                                    True,
                                    0)[0]


class TypeParser(object):
    def __init__(self, atom_tokenizer: AtomTokenizer):
        self.operators = {k for k in atom_tokenizer.atom_map.keys() if k.lower() == k and k != '_'}
        self.operator_classes = {k: BoxType if k in ModDeps else DiamondType for k in self.operators if k != '→'}

    def analyze_beam_batch(self, sents: strs, polishes: List[List[Optional[List[strs]]]]) \
            -> List[Tuple[Tuple[int, int], int, Analysis]]:

        typings: List[List[Optional[OWordTypes]]]
        typings = [[self.sent_to_types(polish) for polish in beam] for beam in polishes]
        polarized: List[List[Optional[OWordTypes]]]
        polarized = [[self.polarize_sent(sent) for sent in beam] for beam in typings]
        atoms_and_indices = [[self.get_atomset_and_indices(sent) for sent in beam] for beam in polarized]
        valid_for_linking = [(s, b) for s in range(len(atoms_and_indices)) for b in range(len(atoms_and_indices[s]))
                             if atoms_and_indices[s][b] is not None]
        proper_analyses = [
            ((s, b),
             len(atoms_and_indices[s][b][0]),
             Analysis(words=sents[s].split(), types=polarized[s][b][1:], conclusion=polarized[s][b][0],
                      polish=atoms_and_indices[s][b][0], atom_set=atoms_and_indices[s][b][1],
                      positive_ids=atoms_and_indices[s][b][2], negative_ids=atoms_and_indices[s][b][3],
                      idx_to_polish=atoms_and_indices[s][b][4]))
            for s, b in valid_for_linking
        ]
        return proper_analyses

    def analyses_to_indices(self, analyses: List[Analysis]) -> Tuple[List[List[LongTensor]], List[List[LongTensor]]]:
        positive_ids, negative_ids = list(zip(*[analysis.get_ids() for analysis in analyses]))
        return tensorize_batch_indexers(positive_ids), tensorize_batch_indexers(negative_ids)

    def polish_to_type(self, polished: strs) -> Optional[WordType]:
        try:
            return polish_to_type(polished, self.operators, self.operator_classes)
        except AssertionError:
            return None
        except IndexError:
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
            -> Optional[Tuple[strs, Atoms, List[ints], List[ints], IntMapping]]:
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


def sample_to_analysis(sample: Sample) -> Analysis:
    words = sample.words
    types = sample.types

    atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, types)))))
    negative, positive = list(map(lambda x: reduce(add, x), atoms))
    conclusion_pair = get_conclusion(positive + negative, sample.proof)
    negative += [conclusion_pair]
    conclusion_type = polarize_and_index(conclusion_pair[0], False, conclusion_pair[1])[1]

    local_atom_set = list(set(map(lambda x: x[0], positive + negative)))
    positive_sep = sep(positive, local_atom_set)
    negative_sep = sep(negative, local_atom_set)

    polished = polish_fn([conclusion_type] + types)
    positional_ids = index_from_polish(polished, offset=-1)
    polish_from_index = {v: k for k, v in positional_ids.items()}

    positive_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                            positive_sep))
    negative_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                            negative_sep))

    return Analysis(words=words, types=types, conclusion=conclusion_type,
                    polish=polished, atom_set=local_atom_set, positive_ids=positive_ids, negative_ids=negative_ids,
                    idx_to_polish=polish_from_index, axiom_links={k: v for k, v in sample.proof})