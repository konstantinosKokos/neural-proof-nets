from Parser.parsing.milltypes import AtomicType

# # # Extraction variables # # #
# Mapping from phrasal categories and POS tags to Atomic Types
_CatDict = {'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'cp': 'CP', 'detp': 'DETP', 'inf': 'INF', 'np': 'NP',
            'oti': 'OTI', 'pp': 'PP', 'ppart': 'PPART', 'ppres': 'PPRES', 'rel': 'REL', 'smain': 'SMAIN',
            'ssub': 'SSUB', 'sv1': 'SV1', 'svan': 'SVAN', 'ti': 'TI', 'whq': 'WHQ', 'whrel': 'WHREL',
            'whsub': 'WHSUB'}
_PosDict = {'adj': 'ADJ', 'adv': 'ADV', 'comp': 'COMP', 'comparative': 'COMPARATIVE', 'det': 'DET',
            'fixed': 'FIXED', 'name': 'NAME', 'noun': 'N', 'num': 'NUM', 'part': 'PART',
            'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'PRON', 'punct': 'PUNCT', 'tag': 'TAG',
            'verb': 'VERB', 'vg': 'VG'}
_PtDict = {'adj': 'ADJ', 'bw': 'BW', 'let': 'LET', 'lid': 'LID', 'n': 'N', 'spec': 'SPEC', 'tsw': 'TSW',
           'tw': 'TW', 'vg': 'VG', 'vnw': 'VNW', 'vz': 'VZ', 'ww': 'WW'}

CatDict = {k: AtomicType(v) for k, v in _CatDict.items()}
PosDict = {k: AtomicType(v) for k, v in _PosDict.items()}
PtDict = {k: AtomicType(v) for k, v in _PtDict.items()}

# Head and modifier dependencies
HeadDeps = frozenset(['hd', 'rhd', 'whd', 'cmp', 'crd', 'det'])
ModDeps = frozenset(['mod', 'predm', 'app'])
