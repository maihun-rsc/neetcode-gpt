from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        tokens = [a for a in corpus]
        merges = []
        for _ in range(num_merges):
        #    a. Count frequency of all adjacent token pairs
            pairs = {} 
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        #    b. Find the most frequent pair (break ties lexicographically)
            merge = sorted(pairs, key=lambda k: (-pairs[k], k))[0]
        #    d. Record the merge as [token_a, token_b]
            merges.append(merge)
        #    c. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1:
                    if (tokens[i], tokens[i+1]) == (merge[0], merge[1]):
                        new_tokens.append(merge[0]+merge[1])
                        i = i + 2
                    else:
                        new_tokens.append(tokens[i])
                        i = i + 1
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        # 3. Return the list of merges performed
        return merges
