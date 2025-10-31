from pgmpy.readwrite import BIFReader
import pandas as pd

reader = BIFReader("networks/child.bif") 
model = reader.get_model()

# sample synthetic data
data = model.simulate(n_samples=10000)

# encode each categorical column to integers based on its unique order
encoded = data.apply(lambda col: pd.Categorical(col).codes)

# compute arities (max code + 1 for each column)
arities = (encoded.max(axis=0) + 1).astype(int).tolist()

out_path = "data/child_10000.dat"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\t".join(encoded.columns) + "\n")     # header
    f.write("\t".join(map(str, arities)) + "\n")   # arities

encoded.to_csv(out_path, sep="\t", index=False, header=False, mode="a")  # data