import pandas as pd
import torch
from torch.utils.data import Dataset

class InEXITDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path).fillna("None")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.r_cols = ["resume_summary", "resume_experience", "resume_skills", "resume_education"]
        self.j_cols = ["jd_overview", "jd_responsibilities", "jd_requirements", "jd_preferred"]

    def __len__(self):
        return len(self.df)

    def _enc(self, text, prefix=""):
        return self.tokenizer.encode(
            prefix + str(text), add_special_tokens=True, 
            max_length=self.max_len, padding='max_length', truncation=True
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        segments = []
        segments.append(self._enc("Resume Content", "passage: ")) # Taxons
        segments.append(self._enc("Job Information", "query: "))
        
        for c in self.r_cols: segments.append(self._enc(c.replace("resume_", ""), "passage: ")) # Keys
        for c in self.j_cols: segments.append(self._enc(c.replace("jd_", ""), "query: "))
        
        for c in self.r_cols: segments.append(self._enc(row[c], "passage: ")) # Values
        for c in self.j_cols: segments.append(self._enc(row[c], "query: "))
        
        label_str = str(row['label']).strip().lower()
        label_val = 1.0 if label_str == 'good fit' else 0.0

        return {"input_ids": torch.tensor(segments), "label": torch.tensor(label_val)}