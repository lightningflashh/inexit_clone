import torch
import torch.nn as nn
from transformers import AutoModel
import copy

class InEXITModel(nn.Module):
    def __init__(self, args):
        super(InEXITModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.pretrained_encoder)
        
        # Mặc định InEXIT dùng 1 layer Transformer cho mỗi giai đoạn
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.word_emb_dim, nhead=8, dim_feedforward=args.word_emb_dim, batch_first=True
        )
        self.internal_encoders = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(args.num_layers)])
        self.external_encoders = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(args.num_layers)])

        self.pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))
        self.mlp = nn.Sequential(
            nn.Linear(args.word_emb_dim * 3, args.word_emb_dim),
            nn.ReLU(),
            nn.Linear(args.word_emb_dim, 1) # Binary classification
        )

    def forward(self, input_ids, attention_mask):
        b_size, num_segs, seq_len = input_ids.shape
        flat_ids = input_ids.view(-1, seq_len)
        flat_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        embeddings = self.pool(outputs).view(b_size, num_segs, -1) 
        
        # Tách segments (2 taxons + 8 keys + 8 values = 18)
        r_taxon = embeddings[:, 0, :]
        j_taxon = embeddings[:, 1, :]
        r_keys, j_keys = embeddings[:, 2:6, :], embeddings[:, 6:10, :]
        r_vals, j_vals = embeddings[:, 10:14, :], embeddings[:, 14:18, :]

        # Phase 1: Inner Interaction (Fusion Key + Value)
        geek = r_keys + r_vals 
        job = j_keys + j_vals
        for enc in self.internal_encoders:
            geek, job = enc(geek), enc(job)

        # Phase 2: Add Taxon info
        geek = geek + r_taxon.unsqueeze(1)
        job = job + j_taxon.unsqueeze(1)

        # Phase 3: External Interaction
        geek_job = torch.cat([geek, job], dim=1)
        for enc in self.external_encoders:
            geek_job = enc(geek_job)

        # Final Representation
        geek_vec, job_vec = torch.split(geek_job, [4, 4], dim=1)
        geek_vec, job_vec = torch.mean(geek_vec, dim=1), torch.mean(job_vec, dim=1)

        # MLP layer: concat(job, geek, job-geek)
        combined = torch.cat([job_vec, geek_vec, job_vec - geek_vec], dim=1)
        return self.mlp(combined).squeeze(1)