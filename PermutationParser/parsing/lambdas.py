from PermutationParser.parsing.milltypes import *
from typing import *
import pickle
from dataclasses import dataclass

IntMapping = Dict[int, int]


with open('./example_analysis.pickle', 'rb') as f:
    analysis = pickle.load(f)[0]


@dataclass
class Node:
    idx: str
    name: str
    polarity: bool
    terminal: bool

    def __init__(self, idx: str, name: str, polarity: bool, terminal: bool):
        self.idx = idx
        self.name = name
        self.polarity = polarity
        self.terminal = terminal

    def __hash__(self):
        return (self.idx, self.name, self.polarity).__hash__()


class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add_atomic_nodes(self, types: WordTypes):
        self.nodes.update(make_atomic_nodes(types))

    def add_intra_graphs(self, types: WordTypes):
        nodes, edges = list(zip(*list(map(make_intra_graphs, types))))
        self.nodes.update(set.union(*nodes))
        self.edges.update(set.union(*edges))

    def add_inter_graphs(self, proofnet: IntMapping):
        self.edges.update({(k, v) for k, v in proofnet.items()})


def make_graph(premises: WordTypes, conclusion: WordType, pnet: IntMapping) -> Graph:
    graph = Graph()
    graph.add_atomic_nodes(premises + [conclusion])
    graph.add_intra_graphs(premises + [conclusion])
    graph.add_inter_graphs(pnet)
    return graph


def get_type_indices(wordtype: WordType) -> List[int]:
    pos, neg = get_polarities_and_indices(wordtype)
    return sorted(reduce(add, map(lambda x: [x[1]], pos), []) + reduce(add, map(lambda x: [x[1]], neg), []))


def make_atomic_nodes(types: WordTypes) -> Set[AtomicType]:
    return set.union(*list(map(get_atomic, types)))


def make_intra_graphs(wordtype: WordType, polarity: bool = True, parent: Optional[str] = None) \
        -> Tuple[Set[Node], Set[Tuple[str, str]]]:
    if isinstance(wordtype, AtomicType):
        return ({Node(idx=wordtype.index, polarity=wordtype.polarity, name=wordtype.type, terminal=True)},
                {(parent, str(wordtype.index))} if parent is not None else set())
    else:
        # get identifiers for each subtree
        left_id = ''.join(list(map(str, get_type_indices(wordtype.argument))))
        right_id = ''.join(list(map(str, get_type_indices(wordtype.result))))
        # init the implication node
        node = Node(idx=f'{left_id} | {right_id}', polarity=polarity, name=get_decoration(wordtype), terminal=False)
        edge = {(parent, node.idx)} if parent is not None else set()
        left_nodes, left_edges = make_intra_graphs(wordtype.argument, not polarity, node.idx)
        right_nodes, right_edges = make_intra_graphs(wordtype.result, polarity, node.idx)
        return {node}.union(left_nodes).union(right_nodes), edge.union(left_edges).union(right_edges)


def get_decoration(functor: FunctorType):
    return functor.diamond if isinstance(functor, DiamondType) else functor.box if isinstance(functor, BoxType) else '→'
