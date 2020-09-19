from LassyExtraction.utils.tools import get_words, get_types, get_name, DAG, WordTypes
from typing import Tuple, List, Set

ProofNet = Set[Tuple[int, int]]
AethelSample = Tuple[DAG, ProofNet]
strs = List[str]


def preprocess_aethel_sample(sample: AethelSample) -> Tuple[strs, WordTypes, ProofNet, str]:
    dag, proof = sample
    return get_words(dag), get_types(dag), proof, get_name(dag)


def preprocess_samples(samples: List[AethelSample]) -> List[Tuple[strs, WordTypes, ProofNet, str]]:
    return [preprocess_aethel_sample(sample) for sample in samples]


def convert_dataset(dataset: List[List[AethelSample]]) -> List[List[Tuple[strs, WordTypes, ProofNet, str]]]:
    return [preprocess_samples(subset) for subset in dataset]
