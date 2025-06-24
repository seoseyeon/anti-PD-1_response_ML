# data processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

## down-load LR pair information
# ligand - receptor pair
lr = pd.read_csv("/Data/CelltalkDB_human_LR.csv")
ligand = lr["ligand_gene_symbol"].tolist()
receptor = lr["receptor_gene_symbol"].tolist()
genes = data.columns.tolist()
idx = lr.index.tolist()

## Data loading
liu = pd.read_csv("/Data/Liu_data.csv")
gide = pd.read_csv("/Data/Gide_data.csv", sep = "\t", index_col= "Unnamed: 0")
riaz = pd.read_csv("/Data/Riaz_data.csv", sep = "\t", index_col= "Unnamed: 0")

## data processing  
def processing(data, metadata, num, log, scale):
    
    if log == "log":
        data = np.log2(data + num)
    
    elif log == 0:
        data = data + num
    
    # colums pairs name
    pair_df = []
    new_df = []
    
    for i in idx:
        
        # ligand & receptor
        ligand = lr.iloc[i, 1]
        receptor = lr.iloc[i, 2]
        
        # value = Ture ? False ?
        if ligand in genes and receptor in genes:
            # ligand_receptor -> column
            pair_name = f"{ligand}_{receptor}"
            pair_df.append(pair_name)
            
            # cal *
            new = pd.DataFrame((data[ligand] * data[receptor]).tolist())
            new_df.append(new)
            
    # result processnig data
    combine = pd.concat(new_df, axis=1)
    
    # minmax or zscore
    
    if scale == 0:
        pass
    
    elif scale == "minmax":
        normalizer = MinMaxScaler()
        normalizer.fit(combine)
        combine = normalizer.transform(combine)
        combine = pd.DataFrame(combine)
    
    elif scale == "zscore":
        normalizer = StandardScaler()
        normalizer.fit(combine)
        combine = normalizer.transform(combine)
        combine = pd.DataFrame(combine)

    combine.columns = pair_df # columns names
    combine.index = data.index.tolist() # patients names
       
    # metadata merge
    result = pd.merge(combine, metadata, left_index=True, right_index=True, how='inner')
       
       
       
    return result

num_list = [0.1]
scale_list = ["minmax"]
condition = list(itertools.product(num_list, scale_list))

# for example
for i in range(0, len(condition)):
    train_data = processing(Liu_data, train_metadata, condition[i][0], 0, condition[i][1])

train_data.to_csv("/Data/train_data.csv")