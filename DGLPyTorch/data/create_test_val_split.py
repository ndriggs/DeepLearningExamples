import pandas as pd
import random
random.seed(42)

df = pd.read_parquet('train.parquet', engine = 'pyarrow')

bb1ids = df.buildingblock1_smiles.unique().tolist()
bb2ids = df.buildingblock2_smiles.unique().tolist()
bb3ids = df.buildingblock3_smiles.unique().tolist()

group2 = list(set(bb2ids) & set(bb3ids))
group3 = list(set(bb3ids) - set(bb2ids))

bbs1 = random.sample(bb1ids, 17)
bbs2 = random.sample(group2, 34)
bbs3 = random.sample(group3, 2)

df['bb_test_noshare'] = False
df.loc[df.buildingblock1_smiles.isin(bbs1) & df.buildingblock2_smiles.isin(bbs2) & df.buildingblock3_smiles.isin(bbs2+bbs3), 'bb_test_noshare'] = True

df['bb_test_mixshare'] = False
df.loc[df.buildingblock1_smiles.isin(bbs1) | \
df.buildingblock2_smiles.isin(bbs2) | df.buildingblock3_smiles.isin(bbs2+bbs3), 'bb_test_mixshare'] \
= True

df['random_test'] = False
random_indices = random.sample(range(len(df)), int(len(df)*0.003))
df.loc[random_indices, 'random_test'] = True

df['full_test'] = df['bb_test_noshare'] + df['bb_test_mixshare'] + df['random_test']
df['full_test'] = df['full_test'] > 0

df.to_parquet('train_val_split.parquet', index=False)