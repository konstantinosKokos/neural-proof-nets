from dataclasses import dataclass
from functools import reduce
from typing import Optional, List, Tuple, overload, Any, Literal

from Parser.data.constants import ModDeps
from Parser.data.preprocessing import (strs, MWU, add, sep, index_from_polish, polish_fn, Atoms, ints,
                                       Sample, make_atom_set, get_conclusion)
from Parser.neural.utils import AtomTokenizer, tensorize_batch_indexers, LongTensor
from Parser.parsing.milltypes import (BoxType, DiamondType, WordType, FunctorType, polish_to_type,
                                      get_polarities_and_indices, polarize_and_index_many,
                                      polarize_and_index, invariance_check)
from Parser.parsing.lambdas import Graph, make_graph, IntMapping, traverse

WordTypes = List[WordType]
OWordType = Optional[WordType]
OWordTypes = List[OWordType]

_atom_set = make_atom_set()


@dataclass(init=False)
class Analysis:
    """
        Class representing a sentence, optionally associated with (1) a linear logic proof frame, (2) a proof structure
        (bijection between atomic formulas), (3) the CHC lambda-term obtained by traversal of the proof structure if
        it is a valid proof net. Property `lambda_term` contains the lambda term including dependency decorations and
        type hints. Property `lambda_term_no_dec` contains the untyped lambda term.
    """
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
    lambda_term_no_dec: Optional[str] = None

    def __init__(self, words: strs, types: Optional[WordTypes], conclusion: Optional[WordType], polish: Optional[strs],
                 atom_set: Optional[Atoms], positive_ids: Optional[List[ints]], negative_ids: Optional[List[ints]],
                 idx_to_polish: Optional[IntMapping], axiom_links: Optional[IntMapping] = None,
                 proof_structure: Optional[Graph] = None, lambda_term: Optional[str] = None,
                 lambda_term_no_dec: Optional[str] = None):
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
        self.lambda_term_no_dec = lambda_term_no_dec

    def __len__(self) -> int:
        """
        :return: The length of atomic types in the proof frame, if one exists, else zero.
        """
        return len(self.polish) if self.polish is not None else 0

    def __repr__(self) -> str:
        """
        :return: A string representation of the proof frame as a linear logic judgement of the form
            `w₀: T₀, ... wₙ: Tₙ ⊢ C`
        """
        return '' if self.types is None or self.conclusion is None else \
            ', '.join([f'{w}: {t}' for w, t in zip(self.words, self.types)]) + f' ⊢ {self.conclusion}'

    def __str__(self) -> str:
        """
        :return: A string representation of the proof frame as a linear logic judgement of the form
            `w₀: T₀, ... wₙ: Tₙ ⊢ C`
        """
        return self.__repr__()

    @overload
    def __eq__(self, other: 'Analysis') -> Optional[bool]:
        pass

    @overload
    def __eq__(self, other: Any) -> Literal[False]:
        pass

    def __eq__(self, other) -> Optional[bool]:
        """
            Equality check between class instance and another object.
        :param other: Object to compare against.
        :return: True, if other is of type Analysis and there is full point-wise agreement between words, types and
            axiom links. None if other is of type Analysis and any of the elements to be compared is None.
            False in all other cases.
        """
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
        words, types = merge_mwus(self.words, self.types)
        self.proof_structure = make_graph(words + [''], types, self.conclusion, False)
        pstruct = make_graph(words + [''], types, self.conclusion, True)

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
        self.lambda_term = traverse(pstruct, str(self.conclusion.index),
                                    {str(k): str(v) for k, v in self.axiom_links.items()},
                                    {str(v): str(k) for k, v in self.axiom_links.items()},
                                    True, 0, add_dependencies=True)[0]
        self.lambda_term_no_dec = traverse(self.proof_structure, str(self.conclusion.index),
                                    {str(k): str(v) for k, v in self.axiom_links.items()},
                                    {str(v): str(k) for k, v in self.axiom_links.items()},
                                    True, 0, add_dependencies=False)[0]


class TypeParser(object):
    def __init__(self, atom_tokenizer: AtomTokenizer):
        self.operators = {k for k in atom_tokenizer.atom_map.keys() if k.lower() == k and k != '_'}
        self.operator_classes = {k: BoxType if k in ModDeps.union({'det'}) else DiamondType
                                 for k in self.operators if k != '→'}
        self.operator_classes['→'] = FunctorType

    def analyze_beam_batch(self, sents: strs, polishes: List[List[Optional[List[strs]]]]) \
            -> List[List[Analysis]]:

        ret = []
        for s in range(len(polishes)):
            words = sents[s].split()
            analyses = []
            for b in range(len(polishes[s])):
                typing: OWordTypes
                polarized: Optional[WordTypes]
                atoms_and_inds: Optional[Tuple[List[str], Atoms, List[List[int]], List[List[int]], IntMapping]]
                valid_for_linking: bool

                typing = self.sent_to_types(polishes[s][b])
                polarized = self.polarize_sent(typing)
                atoms_and_inds = self.get_atomset_and_indices(polarized)
                if atoms_and_inds is None:
                    polish, atom_set, positive_ids, negative_ids, idx_to_polish = None, None, None, None, None
                else:
                    polish, atom_set, positive_ids, negative_ids, idx_to_polish = atoms_and_inds

                analysis = Analysis(words=words, types=polarized[1:] if polarized is not None else None,
                                    conclusion=polarized[0] if polarized is not None else None,
                                    polish=polish,  atom_set=atom_set,
                                    positive_ids=positive_ids, negative_ids=negative_ids, idx_to_polish=idx_to_polish)
                analyses.append(analysis)
            ret.append(analyses)
        return ret

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
        idx = 0
        wordtypes = []
        for wt in sent[1:]:
            if wt == MWU:
                wordtypes.append(wt)
            else:
                idx, wt = polarize_and_index(wt, polarity=True, index=idx)
                wordtypes.append(wt)
        _, conclusion = polarize_and_index(sent[0], polarity=False, index=idx)
        return [conclusion] + wordtypes

    @staticmethod
    def get_atomset_and_indices(sent: Optional[WordTypes]) \
            -> Optional[Tuple[strs, Atoms, List[ints], List[ints], IntMapping]]:
        if sent is None:
            return None
        if not invariance_check(sent[1:], sent[0]):
            return None
        atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, sent[1:])))))
        negative, positive = list(map(lambda x: reduce(add, x), atoms))
        negative += get_polarities_and_indices(sent[0])[1]

        local_atom_set = list(set(map(lambda x: x[0], positive + negative)))
        positive_sep = sep(positive, local_atom_set)
        negative_sep = sep(negative, local_atom_set)

        polished = polish_fn(sent)
        positional_ids = index_from_polish(polished, offset=0)
        polish_from_index = {v: k for k, v in positional_ids.items()}

        positive_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                positive_sep))
        negative_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                negative_sep))
        for p, n in zip(positive_ids, negative_ids):
            if len(p) != len(n):
                return None
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

    lwords, ltypes = merge_mwus(words, types)
    pstruct = make_graph(lwords + [''], ltypes, conclusion_type, True)
    pstruct_nodec = make_graph(lwords + [' '], ltypes, conclusion_type, False)
    lambda_term_dec = traverse(pstruct, str(conclusion_type.index), {str(k): str(v) for k, v in sample.proof},
                               {str(v): str(k) for k, v in sample.proof}, True, 0, add_dependencies=True)[0]
    lambda_term_nodec = traverse(pstruct_nodec, str(conclusion_type.index), {str(k): str(v) for k, v in sample.proof},
                                 {str(v): str(k) for k, v in sample.proof}, True, 0, add_dependencies=False)[0]

    return Analysis(words=words, types=types, conclusion=conclusion_type,
                    polish=polished, atom_set=local_atom_set, positive_ids=positive_ids, negative_ids=negative_ids,
                    idx_to_polish=polish_from_index, axiom_links={k: v for k, v in sample.proof},
                    proof_structure=pstruct, lambda_term=lambda_term_dec, lambda_term_no_dec=lambda_term_nodec)


def merge_mwus(words: List[str], types: WordTypes) -> Tuple[List[str], WordTypes]:
    def get_word(i: int) -> str:
        ret = words[i]
        for j in range(i+1, len(types)):
            if types[j] == MWU:
                ret += f'_{words[j]}'
            else:
                break
        return ret

    owords = [get_word(i) if types[i] != MWU else None for i in range(len(types))]
    words = [w for w in owords if w is not None]
    types = [t for t in types if t != MWU]
    return words, types
