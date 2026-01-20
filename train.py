import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from transformers import AutoTokenizer
from src.model.inexit import InEXITModel
from src.preprocess.dataset import InEXITDataset

class Config:
    pretrained_encoder = "bert-base-multilingual-cased"
    word_emb_dim = 768
    num_layers = 1
    lr = 1e-5
    epochs = 10
    batch_size = 6

def evaluate(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            logits = model(ids, (ids != 0))
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            truths.extend(batch['label'].numpy())
    
    y_pred = [1 if p > 0.5 else 0 for p in preds]
    return {
        "auc": roc_auc_score(truths, preds),
        "acc": accuracy_score(truths, y_pred),
        "f1": f1_score(truths, y_pred),
        "pre": precision_score(truths, y_pred),
        "rec": recall_score(truths, y_pred)
    }

def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_encoder)
    
    train_loader = DataLoader(InEXITDataset("train.csv", tokenizer), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(InEXITDataset("test.csv", tokenizer), batch_size=cfg.batch_size, shuffle=True)

    model = InEXITModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_f1 = 0
    print("--- BẮT ĐẦU TRAINING ---")
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), (batch['input_ids'] != 0).to(device))
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
        
        res = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1} | AUC: {res['auc']:.4f} | F1: {res['f1']:.4f} | ACC: {res['acc']:.4f}")
        
        if res['f1'] > best_f1:
            best_f1 = res['f1']
            torch.save(model.state_dict(), "best_model.pth")

    print("\n--- KẾT QUẢ TEST CUỐI CÙNG (BEST MODEL) ---")
    model.load_state_dict(torch.load("best_model.pth"))
    final_res = evaluate(model, test_loader, device)
    for k, v in final_res.items():
        print(f"{k.upper()}: {v:.4f}")

if __name__ == "__main__":
    main()
