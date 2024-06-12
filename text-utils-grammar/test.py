from text_utils import configuration, tokenization
from sparql_kgqa.sparql.utils import load_sparql_constraint

if __name__ == "__main__":
    cfg = configuration.load_config("../../sparql-kgqa/configs/tokenizers/gpt2.yaml")
    tok = tokenization.Tokenizer.from_config(cfg)
    conts = tok.get_vocab()
    for exact in [False, True]:
        print(f"exact={exact}")
        lr1 = load_sparql_constraint(["wikidata"], conts, exact)
        lr1.reset(b"PREFIX")
        indices, _ = lr1.get()
        print(len(indices))
        print([
            (i, bytes(conts[i]).decode(errors="replace"))
            for i in indices[:10]
        ])
        print()
