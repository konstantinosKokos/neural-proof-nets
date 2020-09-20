from LassyExtraction.utils.tools import get_words, get_types, get_name, DAG, WordTypes
from Parser.data.preprocessing import _atom_set, Sample, preprocess
from typing import Tuple, List, Set
import pickle

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


def main(data_path: str = './train_dev_test.p') -> Tuple[List[Sample], List[Sample], List[Sample]]:
    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
        train, dev, test = convert_dataset(dataset)

    trainsamples = list(filter(lambda x: x is not None,
                               map(lambda x: preprocess(x[0], x[1], x[2], _atom_set, x[3]),
                                   train)))
    devsamples = list(filter(lambda x: x is not None,
                             map(lambda x: preprocess(x[0], x[1], x[2], _atom_set, x[3]),
                                 dev)))
    testsamples = list(filter(lambda x: x is not None,
                              map(lambda x: preprocess(x[0], x[1], x[2], _atom_set, x[3]),
                                  test)))

    with open('./processed.p', 'wb') as f:
        pickle.dump((trainsamples, devsamples, testsamples), f)
        print('Saved pre-processed samples.')
        return trainsamples, devsamples, testsamples
