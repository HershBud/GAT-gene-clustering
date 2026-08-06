# phase1_pca.py
# PCA dimensionality reduction on the Zeisel scRNA dataset
# Output goes to data/output/ for the GAT to pick up in phase 2

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
N_PCS = 50
N_HIGHLY_VARIABLE = 2000


# Load expression matrix
print("Loading expression data...")
expr_df = pd.read_csv(INPUT_DIR / "GSE60361_C1-3005-Expression.txt", sep="\t", index_col=0)
print(f"Raw shape: {expr_df.shape}")  # should be genes * cells

# Transpose
adata = sc.AnnData(X=expr_df.T.values.astype(np.float32),
                   obs=pd.DataFrame(index=expr_df.columns),
                   var=pd.DataFrame(index=expr_df.index))

adata.obs_names_make_unique()
adata.var_names_make_unique()
print(f"adata: {adata.n_obs} cells x {adata.n_vars} genes")

# Load labels and attach
labels = pd.read_csv(INPUT_DIR / "Zeisel_cell_label.csv", index_col=0)
print(f"Label file has {len(labels)} entries")
adata.obs["label"] = labels.reindex(adata.obs_names)["Label"].values
print(adata.obs["label"].value_counts())  # should see 7 clusters


# QC filtering
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=3)
print(f"after QC: {adata.n_obs} cells x {adata.n_vars} genes")


# Normalize + log transform
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)


# Check that normalization didn't do anything weird
print("Max value after log1p:", adata.X.max())
print("Min value after log1p:", adata.X.min())



# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=N_HIGHLY_VARIABLE, flavor="seurat_v3", layer="counts")
n_hvg = adata.var["highly_variable"].sum()
print(f"Highly variable genes selected: {n_hvg}")



# Run PCA
sc.tl.pca(adata, n_comps=N_PCS, use_highly_variable=True)
pca_embeddings = adata.obsm["X_pca"]
print(f"PCA output shape: {pca_embeddings.shape}")  # expecting (3005, 50)-ish
print("Variance explained (first 5 PCs):", adata.uns["pca"]["variance_ratio"][:5])


# Save everything
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "zeisel_pca_embeddings.npy", pca_embeddings)
adata.obs[["label"]].to_csv(OUTPUT_DIR / "zeisel_labels_aligned.csv")
adata.write(OUTPUT_DIR / "zeisel_processed.h5ad")


print(f"\nSaved to {OUTPUT_DIR}/")
print(f"Embeddings: {pca_embeddings.shape}")
print(f"Labels: {len(adata.obs)} cells")
print("Done")
