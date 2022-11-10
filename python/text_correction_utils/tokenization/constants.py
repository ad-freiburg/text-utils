# tokens for marking elements of text that are not regular language tokens
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
SEP = "<sep>"
UNK = "<unk>"
MASK = "<mask>"

SPECIAL_TOKENS = [UNK, BOS, EOS, PAD]

# sep and mask are only used for special purposes, such as decoder only models or masked language modeling
EXTENDED_SPECIAL_TOKENS = SPECIAL_TOKENS + [SEP, MASK]
