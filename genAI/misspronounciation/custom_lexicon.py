from g2p_en import G2p

def load_lexicon(path):
    lex = {}
    with open(path, encoding='utf‑8') as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 2:
                lex[parts[0].lower()] = " ".join(parts[1:])
    return lex

indian_lex = load_lexicon("indian_lexicon.txt")
g2p = G2p()

def g2p_with_indian(word):
    w = word.lower()
    if w in indian_lex:
        return indian_lex[w].split()
    else:
        return g2p(w)

word = "curry"
print(g2p_with_indian(word))
print(g2p(word))