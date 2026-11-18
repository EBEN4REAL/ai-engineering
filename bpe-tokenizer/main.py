
from collections import Counter


class SimpleBPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {}

    def get_pair_counts(self, sequences):
        counts = Counter()

        for sequence in sequences:
            for left, right in zip(
                sequence, sequence[1:]
            ):
                counts[(left, right)] += 1

        return counts

    def merge_pair(self, sequence, pair):
        result = []
        i = 0
        # print("MERGE PAIR => ", pair)
        # print("SEQUENCE => ", sequence)

        while i < len(sequence):
            if (
                i < len(sequence) - 1
                and (sequence[i], sequence[i + 1]) == pair
            ):
                result.append(
                    sequence[i] + sequence[i + 1]
                )
                i += 2
            else:
                result.append(sequence[i])
                i += 1

        return result

    def train(self, text, num_merges=10):
        print("TEXT SPLIT => ", text.split())
        sequences = [
            list(word)
            for word in text.split()
        ]
        # print("sequences => ", sequences)
        self.merges = []

        for _ in range(num_merges):
            counts = self.get_pair_counts(sequences)

            if not counts:
                break

            best_pair, frequency = counts.most_common(1)[0]

            # print(f"Best pair: {best_pair}, Frequency: {frequency}")

            if frequency < 2:
                break

            sequences = [
                self.merge_pair(seq, best_pair)
                for seq in sequences
            ]

            print("sequences => 69", sequences)

            self.merges.append(best_pair)

            print(
                f"Merged {best_pair} "
                f"(frequency: {frequency})"
            )

        # Assign IDs to learned tokens.
        print("sequences => 79", sequences)
        # print("sequence => 80", sequence)
        tokens = sorted({
            token
            for sequence in sequences
            for token in sequence
        })

       

        # Also retain original characters.
        tokens = sorted(
            set(tokens) | set("".join(text.split()))
        )

        print("Tokens => 92", tokens)

        self.vocab = {
            token: index
            for index, token in enumerate(tokens)
        }

    def tokenize(self, text):
        sequence = list(text)

        print(" self.merges => ", self.merges)

        for pair in self.merges:
            sequence = self.merge_pair(
                sequence, pair
            )

        return sequence

    def encode(self, text):
        tokens = self.tokenize(text)

        print(f"Tokens for '{text}': {tokens}")

        result =  [
            self.vocab[token]
            for token in tokens
        ]

        print(f"Token IDs for '{text}': {result}")
        return result


if __name__ == "__main__":
    tokenizer = SimpleBPETokenizer()

    corpus = (
       "payment payment payment "
       "payments payments "
       "paying payer payable"
    )

    tokenizer.train(corpus, num_merges=10)

    print("\nLearned merges:")
    print(tokenizer.merges)

    print("\nTokens:")
    print(tokenizer.tokenize("payments"))

    print("\nToken IDs:")
    print(tokenizer.encode("payments"))
