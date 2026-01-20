import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PJFDataset(Dataset):
    def __init__(self, filepath, args):
        super(PJFDataset, self).__init__()
  
        print(f'\nLoading data from {filepath}')
        self.df = pd.read_csv(filepath)
        
        # Khởi tạo tokenizer từ đường dẫn bert_path trong args
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        self.max_feat_len = args.max_feat_len  
        self.max_sent_len = args.max_sent_len

        self.label_map = {
            "Good Fit": 1.0,
            "No Fit": 0.0
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # 1. Gom các thông tin thuộc tính của Resume (Geek)
        # Sử dụng các cột có sẵn trong CSV của bạn
        resume_metadata = [
            "Summary: " + str(row['resume_summary']),
            "Experience: " + str(row['resume_experience']),
            "Skills: " + str(row['resume_skills']),
            "Education: " + str(row['resume_education'])
        ]

        # 2. Gom các thông tin thuộc tính của Job (JD)
        jd_metadata = [
            "Overview: " + str(row['jd_overview']),
            "Responsibilities: " + str(row['jd_responsibilities']),
            "Requirements: " + str(row['jd_requirements']),
            "Preferred: " + str(row['jd_preferred'])
        ]

        # 3. Tokenize Metadata (Thông tin phân mảnh)
        # BERT sẽ nhận diện đây là một danh sách các câu
        resume_feat_tokens = self.bert_tokenizer(
            resume_metadata, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_feat_len, 
            return_tensors='pt'
        )

        jd_feat_tokens = self.bert_tokenizer(
            jd_metadata, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_feat_len, 
            return_tensors='pt'
        )

        # 4. Tokenize Full Text (Văn bản dài)
        resume_text_tokens = self.bert_tokenizer(
            str(row['resume_text']), 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_sent_len, 
            return_tensors='pt'
        )

        jd_text_tokens = self.bert_tokenizer(
            str(row['job_description_text']), 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_sent_len, 
            return_tensors='pt'
        )

        # Lấy nhãn (label)
        raw_label = row['label']
        # Chuyển từ chữ sang số dùng label_map
        label_value = self.label_map.get(raw_label, 0.0) # Mặc định là 0 nếu không khớp
        label = torch.tensor(label_value)

        # Trả về các tensor cần thiết cho mô hình
        # Squeeze(0) để loại bỏ batch dimension thừa do return_tensors='pt' tạo ra
        return (
            resume_feat_tokens['input_ids'].squeeze(0), 
            resume_feat_tokens['attention_mask'].squeeze(0), 
            jd_feat_tokens['input_ids'].squeeze(0), 
            jd_feat_tokens['attention_mask'].squeeze(0), 
            resume_text_tokens['input_ids'].squeeze(0), 
            resume_text_tokens['attention_mask'].squeeze(0), 
            jd_text_tokens['input_ids'].squeeze(0), 
            jd_text_tokens['attention_mask'].squeeze(0), 
            label
        )

