import SpaGFT as spg
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
save_name = sys.argv[1]
read_name = sys.argv[2]

sc.settings.verbosity = 1
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

import os
# os.system(f"ls {data_folder}")
adata = sc.read_h5ad(read_name)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)

adata.var_names_make_unique()
adata.raw = adata
# QC
sc.pp.filter_genes(adata, min_cells=10)
# Normalization
# sc.pp.normalize_total(adata, inplace=True)
# sc.pp.log1p(adata)

(ratio_low, ratio_high) = spg.gft.determine_frequency_ratio(adata, ratio_neighbors=1,spatial_info="spatial")

spg.calculate_frequcncy_domain(adata, ratio_low_freq = ratio_low, ratio_high_freq = ratio_high, spatial_info="spatial")
adata.write_h5ad(save_name)