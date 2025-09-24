"""
Author: Saleh Refahi
"""

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 
from torch.cuda.amp import  autocast
from torch_geometric.data import Batch
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer

from mole.training.data.datasets import MolDataset
from mole.training.data.utils import open_dictionary

from joblib import Parallel, delayed
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
import subprocess
import wandb
import pickle
import pandas as pd
from rdkit import Chem
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import mean_absolute_error

from util import *
from model import Tuner


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help="Output Directory")
    parser.add_argument('--input', type=str, help="Input Directory")
    parser.add_argument('--exp_name', type=str, help="Experiment Name", default="Triplet_results")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning Rate")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help="Epsilon for Adam optimizer")
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine", choices=["linear", "cosine", "polynomial", "constant"], help="Learning rate scheduler type")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch Size")
    parser.add_argument('--num_epochs', type=int, default=35, help="Number of Epochs")
    parser.add_argument('--n_classes', type=int, default=2, help="Number of Levels")
    parser.add_argument('--pre_trained_model', type=str, default="facebook/esm2_t12_35M_UR50D", help="Path to your LLM model or Huggingface model")
    parser.add_argument('--max_len', type=int, default=1500, help='The Sequence max Length(Tokenizer)')
    parser.add_argument('--order_levels', nargs='+', default=[2,1], type=int, help='A list of order H-levels')
    parser.add_argument('--exclude_easy', type=str2bool, default=False, help='Exclude easy examples')
    parser.add_argument('--batch_hard', type=str2bool, default=False, help='Use hard negative sampling in batches')
    parser.add_argument('--margin', type=float, default=0.9, help='Margin for triplet loss')
    parser.add_argument('--huber_beta', type=float, default=0.25, help='Beta parameter for Huber loss')
    return parser.parse_args()


plt.switch_backend('agg')
plt.clf()

def get_available_gpus(min_free_memory_gb=70):
    available_gpus = []
    try:
        # Convert the required memory to MB
        min_free_memory_mb = min_free_memory_gb * 1024

        # Run nvidia-smi and parse the output
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Process output to find GPUs with enough free memory
        free_memories = result.stdout.strip().split("\n")
        for i, free_memory in enumerate(free_memories):
            if int(free_memory) >= min_free_memory_mb:
                available_gpus.append(i)
                
    except subprocess.CalledProcessError as e:
        print("Error querying GPU memory:", e)
    
    return available_gpus


  # Adjust this as needed
if torch.cuda.is_available():
    available_gpus = get_available_gpus(min_free_memory_gb=65)
    selected_gpu = available_gpus[0] 
    device = torch.device(f'cuda:{selected_gpu}')
    print("Current Deviceeeeee :",device)
else :
    device = torch.device('cpu')
    print("No GPU with sufficient memory found. Using CPU!!!!!!")



# Load the protein tokenizer
protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

def tokenize_protein(seq, max_len=1000):
    """
    Tokenizes a protein sequence and computes a bigram frequency matrix.

    :param seq: Protein sequence string
    :param max_len: Maximum sequence length (after tokenization)
    :param device: Device to place the tensors
    :return: Tuple of (tokenized sequence tensor, bigram matrix numpy array)
    """
    # Tokenize using ESM2 tokenizer
    tokens = protein_tokenizer(seq, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    token_ids = tokens['input_ids'][0].cpu().numpy()
    assert token_ids.shape[0] == max_len, f"Got shape {token_ids.shape[0]}, expected {max_len}"


    # Initialize bigram matrix
    vocab_size = protein_tokenizer.vocab_size  # ESM tokenizer vocab size
    bigram_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    prev_idx = None
    for curr_idx in token_ids:
        if curr_idx == protein_tokenizer.pad_token_id:
            continue  # Skip padding tokens
        if prev_idx is not None:
            bigram_matrix[prev_idx, curr_idx] += 1
        prev_idx = curr_idx

    # Move token_ids to device if needed
    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    return token_tensor, bigram_matrix



# ✅ Load the vocabulary
vocab_path = "vocabulary_207atomenvs_radius0_ZINC_guacamole.pkl"
dictionary = open_dictionary(vocab_path)



def process_record(index, record,max_len):
    """Process a single record."""
    try:
        gene_seq = record["Target"]
        smiles = record["Drug"]

        # Skip if either field is not a string
        if not isinstance(gene_seq, str) or not isinstance(smiles, str):
            raise ValueError("Non-string input")

        embedding,bigram_matrix = tokenize_protein(gene_seq.upper(),max_len)

        try:
            smiles = pd.Series([smiles]).astype("string[pyarrow]")
            x = MolDataset(
                smiles,
                dictionary_inp=dictionary,
                labels=None,
                cls_token=False,
                radius_inp=0,
                useFeatures_inp=False,
                use_class_weights=False,
            )
            embedding_smile = Batch.from_data_list([x[0]])
        except Exception as e:
            print(f"❌ ERROR at index with SMILES: {smiles[0]}")
            print(f"   ↳ Exception: {e}")


        
        return index, np.array(embedding, dtype=np.float16),embedding_smile,bigram_matrix

    except Exception as e:
        logging.error(f"Error processing record at index {index}: {e}")
        return None

def process_batch(records,max_len):
    """Process a batch of records."""
    local_dict = {}
    local_dict_smile = {}
    local_dict_bigram = {}
    for index, record in records.iterrows():
        result = process_record(index, record,max_len)
        if result:
            seq_id, embedding, embedding_smile,bigram_matrix = result
            local_dict[seq_id] = embedding
            local_dict_smile[seq_id] = embedding_smile
            local_dict_bigram[seq_id]=bigram_matrix
    return local_dict, local_dict_smile,local_dict_bigram

def batch_iterator(df, batch_size):
    """Yield batches of size `batch_size` from a DataFrame."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

def is_kekulizable(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return True
    except:
        return False


def read_and_tokenize(file_path, max_len, id2embedding, id2embedding_smile, id2bigram,
                      batch_size=500, n_jobs=32):
    """Read records, tokenize gene sequences and SMILES, and store results."""
        
    # 1) Load and select columns
    records = pd.read_csv(file_path / "train.csv")

    records = records[["Target", "Drug"]].astype(str)

    # 2) Validate SMILES
    valid_mask = records["Drug"].apply(is_kekulizable)

    # 3) Filter, preserve old index, then reset to new 0..N-1
    records = records.loc[valid_mask].copy()
    records["old_index"] = records.index.astype(int)   # store pre-reset index
    records = records.reset_index(drop=True)           # reset index


    	    
    logging.info(f"Processing {len(records)} records in batches from {file_path}...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch,max_len)
        for batch in batch_iterator(records, batch_size)
    )

    # Combine results
    for local_dict, local_dict_smile,local_dict_bigram in results:
        id2embedding.update(local_dict)
        id2embedding_smile.update(local_dict_smile)
        id2bigram.update(local_dict_bigram)

def save_args_to_file(args, order, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_file = output_dir / 'run_params.txt'
    
    with open(args_file, 'w') as f:
        f.write("Run Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write(f"\nReorder levels:\norder: {order}\n")





# ✅ Main dataset class
class DTI_Dataset(Dataset):
    def __init__(self, df, max_len):
        # Pre-filter rows with valid, kekulizable SMILES
        valid_mask = df["Drug"].apply(is_kekulizable)
        self.df = df[valid_mask].reset_index(drop=True)
        self.max_len = max_len
        # Load protein tokenizer
        self.protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smile = row['Drug']
        protein = row['Target']
        y = row['Y']

        # Tokenize protein sequence
        gene_tensor = self.protein_tokenizer(
            protein,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )["input_ids"].squeeze(0)

        # Convert SMILES into graph using MolDataset
        smiles = pd.Series([smile]).astype("string[pyarrow]")
        x = MolDataset(
            smiles,
            dictionary_inp=dictionary,
            labels=None,
            cls_token=False,
            radius_inp=0,
            useFeatures_inp=False,
            use_class_weights=False,
        )

        return gene_tensor, x[0], torch.tensor(y, dtype=torch.float)



def collate_fn(batch):
    gene_tensors, smiles_graphs, labels = zip(*batch)
    gene_batch = torch.stack(gene_tensors)
    smiles_batch = Batch.from_data_list(smiles_graphs)
    label_batch = torch.stack(labels)
    return gene_batch, smiles_batch, label_batch
    



def trainer():
    args = parse_arguments()

    # Access the parameters
    output_path = args.output
    input_path = args.input
    exp_name = args.exp_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    n_classes = args.n_classes
    pre_trained_model = args.pre_trained_model
    weight_decay = args.weight_decay
    adam_epsilon = args.adam_epsilon
    lr_scheduler_type = args.lr_scheduler_type
    warmup_steps = args.warmup_steps

    wandb.init(
        project="FIRM-DTI",  # change this to your actual project name
        name=output_path.split("/")[-1],
        config=vars(args)
    )
        

    # Hyperparameters
    start_overall = time.time()
    SEED = 83
    seed_all(SEED)

    data_dir = Path(input_path)
    log_dir = Path(output_path) / 'trainer'

    #reording the levels
    order = args.order_levels
    reorder(data_dir,order)

    save_args_to_file(args, order, output_path)
    print("Tokenizing.....")
    
    # Initialize the shared dictionary
    id2embedding = dict()
    id2embedding_smile = dict()
    id2bigram=dict()
    graph_embeddings = dict()

    start_time = time.time()
    read_and_tokenize(data_dir ,args.max_len, id2embedding,id2embedding_smile,id2bigram)
    train_time = time.time() - start_time

    records = pd.read_csv(data_dir / "train.csv")
    Y_label = records[records["Drug"].apply(is_kekulizable)]["Y"]
    Y_label = Y_label.reset_index(drop=True)         

    print(f"Time to process training set: {train_time:.6f} seconds")
    print(f"Number of embeddings: {len(id2embedding)}")  
    print(f"Number of graph embeddings: {len(graph_embeddings)}")  



    
    experiment_dir = log_dir / exp_name

    if not experiment_dir.is_dir():
        print("Creating new log-directory: {}".format(experiment_dir))
        experiment_dir.mkdir(parents=True)

    n_bad = 3  # counter for number of epochs that did not improve (counter for early stopping)
    n_thresh = num_epochs  # threshold for number of epochs that did not improve (threshold for early stopping)
    batch_hard = args.batch_hard  # whether to activate batch_hard sampling (recommended)
    exclude_easy = args.exclude_easy  # whether to exclude trivial samples (did not improve performance)
    margin = args.margin  # set this to a float to activate threshold-dependent loss functions (see TripletLoss)
    monitor_plot = True

    # Initialize plotting class (used to monitor loss etc during training)
    pltr = plotter(experiment_dir)

    # Prepare datasets
    datasplitter = DataSplitter(data_dir, id2embedding,id2embedding_smile, n_classes)
    train_splits, val, val_lookup = datasplitter.get_predef_splits()

    val20 = Eval(val_lookup, val, datasplitter, n_classes)

    train = CustomDataset(train_splits, datasplitter, n_classes)
    train_loader = dataloader(train, batch_size)

    # Get the size of the dataset
    dataset_size = len(train_loader.dataset)
    print("Dataset size:", dataset_size)

    model = Tuner(pretrained_model=pre_trained_model)
    model.to(device)

    criterion = TripletLoss(exclude_easy=exclude_easy, batch_hard=batch_hard, margin=margin, n_classes=n_classes)



    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=weight_decay)
    
    
    gradient_accumulation_steps = 1
    
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    
    scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )



    val_set = pd.read_csv(data_dir / "val.csv")

    val_dataset = DTI_Dataset(val_set,args.max_len)

    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False,collate_fn=collate_fn )




    saver = Saver(experiment_dir, n_classes)
    saver.save_checkpoint(model, 0, optimizer, experiment_dir / 'checkpoint.pt')

    print('###### Training parameters ######')
    print('Experiment name: {}'.format(experiment_dir))
    print('LR: {}, BS: {}, free Paras.: {}, n_epochs: {}'.format(learning_rate, batch_size, count_parameters(model), num_epochs))
    print('#############################\n')
    print('Start training now!')

    monitor = init_monitor()
    prev_pcc = 0

    for epoch in tqdm(range(num_epochs)):  # for each epoch

    
        epoch_monitor = init_monitor()
        start = time.time()


        for train_idx, (X_pos_neg, X_anchor, Y, anchor_ids, _, _) in  tqdm(enumerate(train_loader)) : # tqdm(enumerate(islice(train_loader, 200)), total=200):      # 
            
            X_anchor, Y = X_anchor.to(device), Y.to(device)

            

            with autocast(enabled=True): 
            
                anchor, pos, neg,_,_,bdb_pred = model(X_pos_neg, X_anchor)
                

                triplet_loss = criterion(anchor, pos, neg, Y, 0, epoch_monitor, epoch)
       
  
                y_label_tensor = torch.tensor(Y_label.values, dtype=torch.float32, device=bdb_pred.device)
                anchor_idx = torch.as_tensor(anchor_ids, dtype=torch.long, device=bdb_pred.device)
                
               
                # Transform labels to -log10 scale (e.g., pKd)
                y_pKd = -torch.log10(torch.clamp(y_label_tensor[anchor_idx], min=1e-8))
                
                # Compute Huber (smooth L1) loss on the transformed labels
                loss_bdb = torch.nn.functional.smooth_l1_loss(
                    bdb_pred, y_pKd, beta=args.huber_beta
                )
                            
                
                total_loss = triplet_loss +  loss_bdb
    
                
            
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()



                
    
            if train_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]["lr"]

                avg_pos = sum(epoch_monitor["pos"]) / len(epoch_monitor["pos"]) if epoch_monitor["pos"] else 0
                avg_neg = sum(epoch_monitor["neg"]) / len(epoch_monitor["neg"]) if epoch_monitor["neg"] else 0
                separation = avg_neg - avg_pos

                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/triplet_loss": triplet_loss.item(),
                    "train/loss_bdb":loss_bdb.item(),
                    "train/lr": current_lr,
                    "train/avg_pos_dist": avg_pos,
                    "train/avg_neg_dist": avg_neg,
                    "train/separation_margin": separation,
                    "epoch": epoch,
                    "step": epoch * len(train_loader) + train_idx
                })

                
                # --- Evaluation ---
        model.eval()
        all_preds, all_labels = [], []
        model = model.to(device)
        with torch.no_grad():
            for gene_input, smiles_input, y in tqdm(val_loader, desc=f"Evaluating validation set"):
                gene_input = gene_input.to(device)
                with autocast():
                    _,_,pred = model.inference(smiles_input, gene_input)
                    pred = pred.detach().float().cpu().numpy().reshape(-1)
                    label = y.detach().float().cpu().numpy().reshape(-1)
                all_preds.extend(pred)
                all_labels.extend(label)
        
        # --- Convert to numpy arrays ---
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
    
        
        # --- Metrics ---
        pcc, _ = pearsonr(all_preds, all_labels)

        if np.abs(pcc) > prev_pcc :
            prev_pcc = pcc
            saver.save_checkpoint(model, epoch, optimizer, experiment_dir / 'best_checkpoint.pt')

        saver.save_checkpoint(model, epoch, optimizer, experiment_dir / 'checkpoint.pt')
        
        # --- Log to wandb ---
        wandb.log({
            "PCC": pcc,
        })
        model.train()


        train_time = time.time() - start

        # Monitor various metrics during training
        if monitor_plot:
            monitor['loss'].append(sum(epoch_monitor['loss']) / len(epoch_monitor['loss']))
            monitor['norm'].append(sum(epoch_monitor['norm']) / len(epoch_monitor['norm']))
            monitor['pos'].append(sum(epoch_monitor['pos']) / len(epoch_monitor['pos']))
            monitor['neg'].append(sum(epoch_monitor['neg']) / len(epoch_monitor['neg']))
            monitor['min'].append(sum(epoch_monitor['min']) / len(epoch_monitor['min']))
            monitor['max'].append(sum(epoch_monitor['max']) / len(epoch_monitor['max']))
            monitor['mean'].append(sum(epoch_monitor['mean']) / len(epoch_monitor['mean']))
        pltr.plot_distances(monitor['pos'], monitor['neg'])
        pltr.plot_loss(monitor['loss'], file_name='loss.pdf')
        pltr.plot_loss(monitor['norm'], file_name='norm.pdf')
        pltr.plot_minMaxMean(monitor)
        


    end_overall = time.time()
    print(end_overall - start_overall)
    print("Total training time: {:.1f}[m]".format((end_overall - start_overall) / 60))
    return None

if __name__ == '__main__':
    trainer()
