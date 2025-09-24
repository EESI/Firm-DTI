"""
Author: Saleh Refahi
Email: sr3622@drexel.edu
Date: 2024-06-26

"""


import torch
import torch.utils.data
import torch.nn as nn
from transformers import AutoModel
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.utils import to_dense_batch, to_dense_adj

from DeBERTa.deberta.config import ModelConfig
from mole.training.models.mole import AtomEnvEmbeddings
import numpy
from torch_geometric.data import Batch
from torch_geometric.data import Data, Batch
from collections import OrderedDict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Manually define the DeBERTa config (from YAML)
model_config = ModelConfig.from_dict({
    "attention_head_size": 64,
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-07,
    "max_position_embeddings": 0,
    "max_relative_positions": 64,
    "norm_rel_ebd": "layer_norm",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pos_att_type": "p2c|c2p",
    "position_biased_input": False,
    "position_buckets": 0,
    "relative_attention": True,
    "share_att_key": True,
    "type_vocab_size": 0,
    "vocab_size": 211
})


class FiLMProjector(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.scale = nn.Linear(d, d)   # t -> per-dim scale
        self.shift = nn.Linear(d, d)   # t -> per-dim bias
    def forward(self, z_d, z_t):
        γ = torch.sigmoid(self.scale(z_t))  # keep scale bounded
        β = self.shift(z_t)
        return γ * z_d + β                  # drug conditioned on target


class DistanceHead(nn.Module):
    def __init__(self, k=10, sigma=0.2):  # RBF over cosine distance
        super().__init__()
        self.register_buffer("mus", torch.linspace(0, 2, k))
        self.sigma = sigma
        self.fc = nn.Linear(k, 1)
    def forward(self, z_d, z_t):
        z_d = nn.functional.normalize(z_d, dim=-1)
        z_t = nn.functional.normalize(z_t, dim=-1)
        dist = 1- (z_d * z_t).sum(-1)     # cosine sim better!!!                  # [B] in [0,2]
        phi = torch.exp(- (dist.unsqueeze(1) - self.mus)**2 / (2*self.sigma**2))
        y_dist = self.fc(phi).squeeze(-1)                   # [B]
        return y_dist, dist
    

 
class AveragePool1dAlongAxis(nn.Module):
    def __init__(self, axis):
        super(AveragePool1dAlongAxis, self).__init__()
        self.axis = axis

    def forward(self, x, mask=None):
        if mask is not None:
            # Zero out padded tokens
            x = x * mask.unsqueeze(-1).float()
            # Sum and divide by non-zero counts to get the mean of non-padded tokens
            summed = torch.sum(x, dim=self.axis)
            counts = torch.clamp(mask.sum(dim=self.axis, keepdim=True), min=1)  # Avoid division by zero
            return summed / counts
        else:
            # Default mean along the axis
            return torch.mean(x, dim=self.axis)


class Tuner(nn.Module):
    def __init__(self, pretrained_model=""):
        super(Tuner, self).__init__()

        self.avg_pooling = AveragePool1dAlongAxis(1)  # Averaging layer
        # Load the pre-trained model
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model)
    
        SCRIPT_DIR = Path(__file__).resolve().parent
        model_PATH = SCRIPT_DIR / "./MolE_GuacaMol_27113.ckpt"
        
        
        self.smile_model = AtomEnvEmbeddings(model_config)        
        
                
        torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
        
        # Now load the checkpoint safely
        ckpt = torch.load(model_PATH, map_location=device, weights_only=False)

        raw_state_dict = ckpt['models_state_dict'][0]


        # List any prefixes you want to remove
        prefixes_to_remove = ("MolE.")  # add more if needed

        clean_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            new_k = k
            for p in prefixes_to_remove:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]  # strip it once
            clean_state_dict[new_k] = v



        # Load into model
        self.smile_model.load_state_dict(clean_state_dict, strict=False)        
                

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        for param in self.pretrained_model.encoder.layer[-3:].parameters():
            param.requires_grad = True 

 
        for param in self.smile_model.parameters():
            param.requires_grad = False

        for param in self.smile_model.encoder.layer[-3:].parameters():
            param.requires_grad = True                
            

        self.embedding_size_1 = self.pretrained_model.config.hidden_size
        self.embedding_size_2 = self.smile_model.config.hidden_size
            
            
        # projection layers for different input sizes
        self.proj_1 = nn.Linear(self.embedding_size_1, 256)
        self.proj_2 = nn.Linear(self.embedding_size_2, 256)
        
        # shared layers
        self.shared_fc2 = nn.Linear(256, 256)
        self.shared_fc3 = nn.Linear(256, 256)
        self.shared_fc4 = nn.Linear(256, 256)
      
        # shared subnetwork (excluding the input projection)
        self.shared_tuner = nn.Sequential(
            nn.ReLU(),
            self.shared_fc2,
            nn.ReLU(),
            self.shared_fc3,
            nn.ReLU(),
            self.shared_fc4,
        )

        
        self.Tuner_1 = nn.Sequential(
            *self.shared_tuner  
        )        
        


        # keep towers (ESM/MolE) and projections
        self.film = FiLMProjector(d=256)
        self.aff_head = DistanceHead(k=10, sigma=0.2)  # distance-only predictor



        
        
    def single_pass3(self, embed):
        pooled_output = self.shared_tuner(embed)
        return pooled_output


            
    def single_pass2(self, data_batch):
        device = next(self.parameters()).device
    
        if isinstance(data_batch, Batch):
            data_batch = data_batch.to(device)
        else:
            data_batch = Batch.from_data_list(data_batch).to(device)
    
        input_ids, attention_mask = to_dense_batch(data_batch.x, data_batch.batch, fill_value=0)
        # attention_mask = bool_mask.long()                            # <- long 1/0
    
        relative_pos = to_dense_adj(
            data_batch.edge_index, data_batch.batch, data_batch.edge_attr
        )
    
        outputs = self.smile_model(
            input_ids=input_ids.to(device),
            input_mask=attention_mask.to(device),
            relative_pos=relative_pos.to(device)
        )
    
        last_hidden_state = outputs["hidden_states"][-1].contiguous()  # [B, L, H]
        token_proj = self.proj_2(last_hidden_state)                     # [B, L, 256]
        avg = self.avg_pooling(token_proj, mask=attention_mask)         # [B, 256]
    
        self.Tuner_1.to(data_batch.x.device)  # <- remove (see Fix #2)
        pooled_output = self.Tuner_1(avg.contiguous())
        return pooled_output
               
     

    def single_pass(self, X):
        X = X.long()
        attention_mask = (X != 1).int()          # <- long(), not int()
        self.pretrained_model = self.pretrained_model.to(X.device)
        attention_mask = attention_mask.to(X.device)
    
        # IMPORTANT: pass mask to HF model
        outputs = self.pretrained_model(X)
        last_hidden_state = outputs.last_hidden_state               # [B, L, H]
    
        token_proj = self.proj_1(last_hidden_state)                 # [B, L, 256]
        avg = self.avg_pooling(token_proj, mask=attention_mask)     # [B, 256]
    
        # Avoid moving modules inside forward (see Fix #2 below)
        self.Tuner_1.to(X.device)  # <- remove
        pooled_output = self.Tuner_1(avg)                           # [B, 256]
        return pooled_output, 0





    def forward(self, X_pos_neg, X_anchor):
            
        pos_batch_list = [item[0] for item in X_pos_neg]
        neg_batch_list = [item[1] for item in X_pos_neg]

        z_t, _ = self.single_pass(X_anchor.contiguous())  # target: [B,256]
        z_pos   = self.single_pass2(pos_batch_list)       # drug+: [B,256]
        z_neg   = self.single_pass2(neg_batch_list)       # drug-: [B,256]

        # condition drug on its target
        z_pos = self.film(z_pos, z_t)
        z_neg = self.film(z_neg, z_t)

        # distance-only affinity prediction for positives
        y_hat, dist_pos = self.aff_head(z_pos, z_t)

        # return distances for your triplet/ranking if you use them
        return z_t, z_pos, z_neg,  dist_pos , dist_pos, y_hat 
    


    def inference(self, X_pos, X_anchor):
        z_t, _  = self.single_pass(X_anchor.contiguous())
        z_pos   = self.single_pass2(X_pos)
        z_pos = self.film(z_pos, z_t)
        y_hat, _ = self.aff_head(z_pos, z_t)
        return z_t,z_pos,y_hat
   

  
        
        
        
        
        
