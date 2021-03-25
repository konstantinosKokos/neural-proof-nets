from dataclasses import dataclass

from LassyExtraction.milltypes import (polish_to_type, WordType, polarize_and_index,  AtomicType, invariance_check,
                                       get_polarities_and_indices, unarize_polish)
from LassyExtraction.aethel import ProofNet, ProofFrame, Premise

from ..neural.utils import tensorize_batch_indexers
from ..data.preprocessing import (MWU, separate, idx_from_polish, reduce, add, polish_seq, pad_mwus, Sample,
                                  remove_polarities)

from typing import Dict, Optional

from torch import Tensor


class ParseError(Exception):
    def __init__(self, message: str):
        super(ParseError, self).__init__(message)


class TypeParser:
    @staticmethod
    def polish_to_type(symbols: list[str]) -> WordType:
        try:
            return polish_to_type(unarize_polish(symbols))
        except (AssertionError, IndexError):
            raise ParseError(f'Could not parse sequence: {symbols}')

    def polishes_to_types(self, symbols: list[list[str]]) -> list[WordType]:
        if not symbols:
            raise ParseError(f'Non-iterable sequence: {symbols}')
        return [self.polish_to_type(s) for s in symbols]

    @staticmethod
    def polarize_types(types: list[WordType]) -> list[WordType]:
        idx = 0
        ret = []
        for t in types[1:]:
            if t != MWU:
                idx, t = polarize_and_index(t, True, idx)
            ret.append(t)
        _, conclusion = polarize_and_index(types[0], False, idx)
        return [conclusion] + ret

    @staticmethod
    def get_atomset_and_indices(types: list[WordType]) \
            -> tuple[list[str], list[AtomicType], list[list[int]], list[list[int]], Dict[int, int]]:

        try:
            if not invariance_check([t for t in types[1:] if t != MWU], types[0]):
                raise ParseError(f'Failing to satisfy invariance for judgement {types[1:]} ˫ {types[0]}')
        except ValueError:
            raise ParseError(f'Failing to satisfy invariance for judgement {types[1:]} ˫ {types[0]}')

        atoms = list(zip(*list(map(get_polarities_and_indices, filter(lambda wordtype: wordtype != MWU, types[1:])))))
        negative, positive = list(map(lambda x: reduce(add, x), atoms))
        negative += get_polarities_and_indices(types[0])[1]

        local_atom_set = list(set(map(lambda x: x[0], positive + negative)))
        positive_sep = separate(positive, local_atom_set)
        negative_sep = separate(negative, local_atom_set)

        polished = polish_seq(types)
        positional_ids = idx_from_polish(polished, offset=0)
        polish_from_index = {v: k for k, v in positional_ids.items()}

        positive_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                positive_sep))
        negative_ids = list(map(lambda idxs: list(map(lambda atom: positional_ids[atom[1]], idxs)),
                                negative_sep))
        for p, n in zip(positive_ids, negative_ids):
            if len(p) != len(n):
                raise ParseError(f'Uneven positives and negatives {p}, {n}')
        return remove_polarities(polished), local_atom_set, positive_ids, negative_ids, polish_from_index

    def parse_beam_batch(self, sents: list[str], decoder_output: list[list[Optional[list[list[str]]]]]):
        ret: list[list[Analysis]] = []

        for sent, symbol_seqs in zip(sents, decoder_output):
            analyses: list[Analysis] = []
            for symbol_seq in symbol_seqs:
                analysis = Analysis()
                try:
                    analysis.words = sent.split()
                    analysis.types = self.polarize_types(self.polishes_to_types(symbol_seq))
                    analysis.polish, analysis.atom_set, analysis.pos_ids, analysis.neg_ids, analysis.idx_to_polish = \
                        self.get_atomset_and_indices(analysis.types)
                except ParseError as e:
                    analysis.traceback = e
                analyses.append(analysis)
            ret.append(analyses)
        return ret


@dataclass
class Analysis:
    words: list[str] = None
    types: list[WordType] = None
    polish: list[str] = None
    atom_set: list[AtomicType] = None
    pos_ids: list[list[int]] = None
    neg_ids: list[list[int]] = None
    idx_to_polish: Dict[int, int] = None
    link_weights: list[list[list[float]]] = None
    axiom_links: Dict[int, int] = None
    traceback: ParseError = None

    def valid(self) -> bool:
        return self.polish is not None

    @staticmethod
    def to_indices(analyses: list['Analysis']) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        p_ids, n_ids = list(zip(*[(analysis.pos_ids, analysis.neg_ids) for analysis in analyses]))
        return tensorize_batch_indexers(p_ids), tensorize_batch_indexers(n_ids)

    def fill_matches(self, matches: list[list[int]]) -> None:
        self.axiom_links = dict()
        for pos, neg, match in zip(self.pos_ids, self.neg_ids, matches):
            for i, p in enumerate(pos):
                n_idx = match[i]
                n = neg[n_idx]
                self.axiom_links[self.idx_to_polish[p]] = self.idx_to_polish[n]
            if len(set(self.axiom_links.keys())) != len(set(self.axiom_links.values())):
                self.traceback = ParseError('Disconnected graph.')

    def to_proofnet(self, name: Optional[str] = None) -> ProofNet:
        return ProofNet(self.to_proof_frame(), axiom_links={(k, v) for k, v in self.axiom_links.items()}, name=name)

    def to_proof_frame(self) -> ProofFrame:
        words, types = merge_mwus(self.words, self.types[1:])
        return ProofFrame(premises=[Premise(word, wt) for word, wt in zip(words, types)], conclusion=self.types[0])

    @staticmethod
    def from_sample(sample: Sample) -> 'Analysis':
        words = sample.words
        types = sample.types
        polish = sample.polish
        axiom_links = {k: v for k, v in sample.proof}
        pos_ids = [pid for pid in sample.pos_ids if pid]
        neg_ids = [nid for nid in sample.neg_ids if nid]
        return Analysis(words=words, types=types, polish=polish, axiom_links=axiom_links,
                        pos_ids=pos_ids, neg_ids=neg_ids)


def merge_mwus(words: list[str], types: list[WordType]) -> tuple[list[str], list[WordType]]:
    def get_word(i: int) -> str:
        ret = words[i]
        for j in range(i+1, len(types)):
            if types[j] == MWU:
                ret += f' {words[j]}'
            else:
                break
        return ret

    owords = [get_word(i) if types[i] != MWU else None for i in range(len(types))]
    words = [w for w in owords if w is not None]
    types = [t for t in types if t != MWU]
    return words, types
