import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import copy

class BertMatchingModel(nn.Module):
    def __init__(self, args):
        super(BertMatchingModel, self).__init__()
        self.args = args
        self.bert = AutoModel.from_pretrained(args.bert_path)
        
        # Cho phép update trọng số BERT
        for param in self.bert.parameters():
            param.requires_grad = True

        # Khởi tạo các lớp Encoder (Transformer)
        # word_emb_dim thường là 768 với BERT base
        self.encoder = Encoder(args.word_emb_dim, args.num_heads, args.hidden_size, args.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(args.num_layers)])

        self.encoder_2 = Encoder(args.word_emb_dim, args.num_heads, args.hidden_size, args.dropout)
        self.encoders_2 = nn.ModuleList([copy.deepcopy(self.encoder_2) for _ in range(args.num_layers)])

        # Pooling để lấy vector đại diện
        self.pool = nn.AdaptiveAvgPool2d((1, args.word_emb_dim))

        # Lớp dự đoán cuối cùng
        self.mlp = nn.Sequential(
            nn.Linear(args.word_emb_dim * 3, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, 1)
        )

    def get_bert_embedding(self, input_ids, attention_mask):
        """Hàm hỗ trợ lấy embedding từ BERT và pooling"""
        # input_ids shape: [batch_size, seq_len] hoặc [batch_size, num_fields, seq_len]
        if len(input_ids.shape) == 3:
            batch_size, num_fields, seq_len = input_ids.shape
            # Flatten để đưa vào BERT một lượt cho nhanh
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)
            
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # last_hidden_state: [batch_size * num_fields, seq_len, 768]
            pooled = self.pool(outputs.last_hidden_state).view(batch_size, num_fields, -1)
            return pooled # [batch_size, num_fields, 768]
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.pool(outputs.last_hidden_state).squeeze(1)
            return pooled.unsqueeze(1) # [batch_size, 1, 768]

    def forward(self, resume_feat_ids, resume_feat_mask, jd_feat_ids, jd_feat_mask, 
                resume_text_ids, resume_text_mask, jd_text_ids, jd_text_mask):
        
        # 1. Lấy embedding cho các trường Metadata (4 trường mỗi bên)
        # Shape trả về: [batch, 4, 768]
        resume_feat_emb = self.get_bert_embedding(resume_feat_ids, resume_feat_mask)
        jd_feat_emb = self.get_bert_embedding(jd_feat_ids, jd_feat_mask)

        # 2. Lấy embedding cho văn bản dài (1 trường mỗi bên)
        # Shape trả về: [batch, 1, 768]
        resume_text_emb = self.get_bert_embedding(resume_text_ids, resume_text_mask)
        jd_text_emb = self.get_bert_embedding(jd_text_ids, jd_text_mask)

        # 3. Kết hợp các trường lại thành chuỗi sequence để đưa vào Transformer
        # Mỗi bên sẽ có 5 "tokens" (4 metadata + 1 full text)
        resume_seq = torch.cat([resume_feat_emb, resume_text_emb], dim=1) # [batch, 5, 768]
        jd_seq = torch.cat([jd_feat_emb, jd_text_emb], dim=1)           # [batch, 5, 768]

        # 4. Inner interaction: Resume tự học, Job tự học
        for encoder in self.encoders:
            resume_seq = encoder(resume_seq)
            jd_seq = encoder(jd_seq)

        # 5. Cross interaction: CV và JD tương tác với nhau
        combined_seq = torch.cat([resume_seq, jd_seq], dim=1) # [batch, 10, 768]
        for encoder_2 in self.encoders_2:
            combined_seq = encoder_2(combined_seq)

        # 6. Tách ra và lấy vector đại diện cuối cùng cho mỗi bên
        resume_final, jd_final = torch.split(combined_seq, 5, dim=1)
        resume_vec = torch.mean(resume_final, dim=1) # [batch, 768]
        jd_vec = torch.mean(jd_final, dim=1)         # [batch, 768]

        # 7. Tính toán đặc trưng kết hợp (Concat, Difference)
        # Đây là kỹ thuật phổ biến trong Matching (giống mô hình InferSent/ESIM)
        combined_features = torch.cat([resume_vec, jd_vec, resume_vec - jd_vec], dim=1)
        
        # 8. Dự đoán kết quả
        output = self.mlp(combined_features).squeeze(1) # [batch]
        return output

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


# --- Các lớp phụ trợ ---

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5 
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x 
        out = self.layer_norm(out)
        return out


