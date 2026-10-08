# %%

import pandas as pd
# %%
train_p = "/home/guests/leonhard_feiner/source/stochastic_segmentation_networks/assets/train_info.csv"
valid_p = "/home/guests/leonhard_feiner/source/stochastic_segmentation_networks/assets/valid_info.csv"
p = train_p
df_orig = pd.read_csv(p)

# %%
id_col = df_orig[["Grade", "BraTS_2018_subject_ID"]].dropna().apply("/".join, axis=1)
id_col
# %%
from operator import itemgetter
valid_orig = pd.read_csv("/home/guests/leonhard_feiner/source/stochastic_segmentation_networks/assets/data_valid_orig.csv")
orig_id = valid_orig.id.str.split("/").apply(itemgetter(-1))


result = pd.merge(orig_id, df_orig, left_on="id", right_on="BraTS_2018_subject_ID", how="left")
# %%
orig_id.shape, df_orig.shape, result.shape
# %%
modalities = "seg", "flair", "t1", "t1ce", "t2"

for modality
modality="flair"
result.BraTS_2020_subject_ID.apply("~/datasets_own/BraTS/MICCAI_BraTS2020_TrainingData/{x}/{x}_{modality}.nii.gz".format)