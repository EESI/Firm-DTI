from tdc.benchmark_group import dti_dg_group
import pandas as pd 


# --- Load Benchmark ---
group = dti_dg_group(path='./TDC_data/')

benchmark = group.get('BindingDB_Patent')
name = benchmark['name']
train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=0)
test = benchmark['test']

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)
valid.to_csv("val.csv",index=False)


df.to_csv("ref.txt", sep="\t", index=True,header=None)

