from hypothesis import given
from hypothesis.strategies import *
from Parser.data.preprocessing import Matrix, Matrices, ProofNet, reduce, add, convert_matches_to_matrix
from typing import List, Tuple


def test_bistochasticity(matrix: Matrix) -> None:
    if not matrix:
        return

    # assert same len rows
    lens = set(map(len, matrix))
    assert len(lens) == 1
    # assert squareness
    assert len(matrix) == list(lens)[0]
    # assert exactly 1 true element per row
    assert all(list(map(lambda row: len(list(filter(lambda el: el, row))) == 1, matrix)))
    transpose = list(map(list, zip(*matrix)))
    # assert at least 1 true element per col
    assert all(list(map(lambda row: len(list(filter(lambda el: el, row))) == 1, transpose)))


def test_support(_matrices: Matrices, proof: ProofNet) -> None:
    assert reduce(add, map(len, _matrices)) == len(proof)


@composite
def ranges(draw, size: int) -> range:
    stop = draw(integers(0, size))
    if stop == 0:
        stop = 2
    if stop % 2:
        stop += 1
    return range(0, stop)


@composite
def proofnets(draw, size: int) -> ProofNet:
    _range = draw(ranges(size))
    perm = draw(permutations(list(_range)))
    pos = perm[0:len(_range)//2]
    neg = perm[len(_range)//2:]
    return list(zip(pos, neg))


@composite
def bins(draw, n_elements: int, n_bins: int) -> List[Tuple[int, int]]:
    start = 0
    out = []

    for n_bin in range(n_bins):
        stop = draw(integers(min_value=start, max_value=n_elements))
        stop = n_elements if n_bin + 1 == n_bins else stop
        out.append((start, stop))
        start += stop - start
    return out


@composite
def matrices(draw, size: int) -> Tuple[Matrices, ProofNet]:
    proofnet = draw(proofnets(size))
    proofnet = draw(permutations(proofnet))
    n_atoms = draw(integers(min_value=1, max_value=20))
    _bins = draw(bins(len(proofnet), n_atoms))
    binned = [proofnet[_bin[0]:_bin[1]] for _bin in _bins]
    positives = [sorted([x[0] for x in _bin]) for _bin in binned]
    negatives = [sorted([x[1] for x in _bin]) for _bin in binned]
    _matrices = list(map(lambda pos, neg: convert_matches_to_matrix((pos, neg), proofnet),
                         negatives,
                         positives))

    return _matrices, proofnet


@given(matrices(50))
def assert_matrix_correctness(x: Tuple[Matrices, ProofNet]) -> None:
    _matrices, proof = x
    for matrix in _matrices:
        test_bistochasticity(matrix)
    test_support(_matrices, proof)
