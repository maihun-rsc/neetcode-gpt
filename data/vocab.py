from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        stoi, itos = {}, {}
        text = sorted(set(text))
        
        stoi = {ch: i for i, ch in enumerate(text)}
        itos = {i: ch for ch, i in stoi.items()}
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        return stoi, itos
        pass

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        encoding = []
        for x in text:
            encoding.append(stoi[x])
        return encoding

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        string = ""
        for i in ids:
            string += itos[i]
        return string
