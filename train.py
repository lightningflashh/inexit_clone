# coding: UTF-8
import os
import sys
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings

from dataset import PJFDataset 
from model import BertMatchingModel
from utils import parse_args, get_logger, classify, keep_only_the_best, get_parameter_number

warnings.filterwarnings('ignore')

def main():
    args = parse_args()

    # 1. Khởi tạo Logger và Thư mục lưu trữ
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    log_path = os.path.join(args.log_dir, f'{args.token}-seed{args.seed}.log')
    logger = get_logger(log_path)
    logger.info(args)
    
    # 2. Thiết lập Seed và Device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    # 3. Chuẩn bị DataLoaders (Sửa để đọc trực tiếp file CSV)
    # Giả sử args.data_path là thư mục chứa train.csv, valid.csv, test.csv
    # 3.1. Đọc file train thô
    df_all = pd.read_csv(os.path.join(args.data_path, 'train.csv'))

    # 3.2. Chia tập dữ liệu có phân tầng (stratify)
    # Tham số stratify=df_all['label'] giúp giữ nguyên tỉ lệ nhãn No Fit / Good Fit
    train_df, valid_df = train_test_split(
        df_all, 
        test_size=0.2, 
        random_state=args.seed, 
        stratify=df_all['label'] 
    )

    # 3.3. Lưu ra file tạm để Dataset nạp vào
    train_path = os.path.join(args.data_path, 'train_tmp.csv')
    valid_path = os.path.join(args.data_path, 'valid_tmp.csv')
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)

    # 3.4. Khởi tạo Dataset từ file tạm
    train_dataset = PJFDataset(train_path, args)
    valid_dataset = PJFDataset(valid_path, args)
    
    test_dataset = PJFDataset(os.path.join(args.data_path, 'test.csv'), args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True)

    # 4. Khởi tạo Model, Loss và Optimizer
    model = BertMatchingModel(args).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    logger.info(get_parameter_number(model))

    # 5. Hàm tính Metrics
    def compute_metrics(text, labels, preds, loss):
        auc = roc_auc_score(labels, preds)
        # Chuyển logits sang nhãn 0/1 để tính Acc/P/R/F1
        preds_binary = (torch.sigmoid(torch.tensor(preds)) > 0.5).numpy()
        TP, FN, FP, TN = classify(labels, preds_binary)
        
        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        info = f"{text} Loss: {loss:.4f} | AUC: {auc:.4f} | ACC: {acc:.4f} | F1: {f1:.4f}"
        logger.info(info)
        return acc, auc, precision, recall, f1

    # 6. Vòng lặp huấn luyện chính
    best_acc = 0
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            # Giải nén 9 items trả về từ PJFDataset
            (res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
             res_t_ids, res_t_mask, jd_t_ids, jd_t_mask, labels) = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
                            res_t_ids, res_t_mask, jd_t_ids, jd_t_mask)
            
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient_norm)
            optimizer.step()

            train_loss += loss.item()
            all_preds += outputs.detach().cpu().tolist()
            all_labels += labels.cpu().tolist()

        compute_metrics(f"Epoch {epoch+1} Train", all_labels, all_preds, train_loss/len(train_loader))

        # 7. Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                (res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
                 res_t_ids, res_t_mask, jd_t_ids, jd_t_mask, labels) = [x.to(device) for x in batch]
                
                outputs = model(res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
                                res_t_ids, res_t_mask, jd_t_ids, jd_t_mask)
                val_loss += criterion(outputs, labels).item()
                all_preds += outputs.cpu().tolist()
                all_labels += labels.cpu().tolist()

        acc, auc, prec, rec, f1 = compute_metrics(f"Epoch {epoch+1} Val", all_labels, all_preds, val_loss/len(valid_loader))

        # Lưu checkpoint
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save({'net': model.state_dict()}, os.path.join(args.save_path, 'model-best.pth.tar'))
            logger.info(f"--- New Best Model at Epoch {best_epoch} ---")

        # Early Stopping
        if epoch + 1 >= best_epoch + args.end_step:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # 8. Testing (Sử dụng model tốt nhất)
    logger.info("Starting Testing...")
    model.load_state_dict(torch.load(os.path.join(args.save_path, 'model-best.pth.tar'))['net'])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            (res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
             res_t_ids, res_t_mask, jd_t_ids, jd_t_mask, labels) = [x.to(device) for x in batch]
            outputs = model(res_f_ids, res_f_mask, jd_f_ids, jd_f_mask, 
                            res_t_ids, res_t_mask, jd_t_ids, jd_t_mask)
            all_preds += outputs.cpu().tolist()
            all_labels += labels.cpu().tolist()
    
    compute_metrics("Final Test", all_labels, all_preds, 0)

if __name__ == '__main__':
    main()

