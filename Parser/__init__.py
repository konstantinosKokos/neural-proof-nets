import Parser
import sys
from Parser.parsing import milltypes

# backwards comp for pickles
sys.modules['PermutationParser'] = Parser
sys.modules['LassyExtraction.milltypes'] = milltypes
