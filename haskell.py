from datasets import load_dataset

ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Haskell"])

print(next(iter(ds)))