# coding: UTF-8
import os
import sys
import datetime
import logging
import platform
import argparse

try:
    import opts as opt
except ImportError:
    opt = None

def parse_args():
    """ Phân tích các tham số đầu vào """
    parser = argparse.ArgumentParser(  
        description='BERT Matching Training', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    
    if opt:
        opt.add_md_help_argument(parser) 
        opt.JRM_opts(parser)  
    else:
        # Nếu bạn không có file opts.py, hãy định nghĩa cơ bản ở đây
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--gpu', type=int, default=0)
        # ... thêm các tham số khác nếu cần
        
    return parser.parse_args()

def get_parameter_number(net):
    """ Thống kê số lượng tham số của mô hình """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def keep_only_the_best(args, best_epoch):
    """ Sao lưu file model tốt nhất """
    best_file_path = os.path.join(args.save_path, f'{args.token}-seed{args.seed}-model-best.pth.tar')
    ori_path = os.path.join(args.save_path, f'{args.token}-seed{args.seed}-model-{best_epoch}.pth.tar')
    
    if os.path.exists(ori_path):
        cmd = 'copy' if platform.system() == "Windows" else 'cp'
        os.system(f'{cmd} {ori_path} {best_file_path}')

def get_logger(log_file=None):
    """ Khởi tạo hệ thống ghi nhật ký (Logging) """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Xóa các handler cũ nếu có để tránh lặp log
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

def classify(ans, pre, threshold=0.5):
    """ 
    Tính toán Confusion Matrix 
    ans: nhãn thật (0/1)
    pre: xác suất dự đoán hoặc nhãn dự đoán sau threshold
    """
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(ans)):
        actual = int(ans[i])
        # Nếu pre là xác suất (float), so sánh với threshold
        prediction = 1 if pre[i] >= threshold else 0
        
        if actual == 1:
            if prediction == 1: TP += 1
            else: FN += 1
        else:
            if prediction == 1: FP += 1
            else: TN += 1
    return TP, FN, FP, TN

