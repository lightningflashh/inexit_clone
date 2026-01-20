from __future__ import print_function
import argparse

def JRM_opts(parser):
    """
    Các tùy chọn cho khởi tạo mô hình và huấn luyện.
    """

    # --- Nhóm thiết lập chung ---
    group = parser.add_argument_group('JRM init')
    group.add_argument('-token', type=str, default='BERT-Match',
                       help="Tên định danh cho phiên bản train này")
    group.add_argument('-data_path', default="data",
                       help="Thư mục chứa các file train.csv, valid.csv, test.csv")
    group.add_argument('-bert_path', default="bert-base-uncased",
                       help="Đường dẫn đến model BERT (local hoặc huggingface hub)")
    group.add_argument('-save_path', default="save",
                       help="Thư mục lưu model checkpoint")
    group.add_argument('-log_dir', default="log",
                       help="Thư mục lưu log file")
    group.add_argument('-gpu', type=int, default=0,
                       help="ID của GPU (nếu dùng CPU thì để mặc định)")

    # --- Nhóm thiết lập huấn luyện ---
    group = parser.add_argument_group('Training setup')
    group.add_argument('-train_batch_size', type=int, default=16)
    group.add_argument('-valid_batch_size', type=int, default=16)
    group.add_argument('-test_batch_size', type=int, default=16)
    
    group.add_argument('-learning_rate', type=float, default=2e-5,
                       help="LR cho fine-tuning BERT thường từ 2e-5 đến 5e-5")
    group.add_argument('-weight_decay', type=float, default=1e-4)
    group.add_argument('-max_gradient_norm', type=float, default=1.0)
    
    # Số lượng trường Metadata (Sửa lại cho khớp với CSV: Summary, Experience, Skills, Education)
    group.add_argument('-geek_max_feature_num', type=int, default=4) 
    group.add_argument('-job_max_feature_num', type=int, default=4) 
    
    group.add_argument('-max_feat_len', type=int, default=64,
                       help='Độ dài tối đa cho mỗi trường metadata')
    group.add_argument('-max_sent_len', type=int, default=256,
                       help='Độ dài tối đa cho văn bản full-text (resume_text/jd_description)') 
    
    group.add_argument('-word_emb_dim', type=int, default=768,
                       help='768 cho BERT base, 1024 cho BERT large') 
    group.add_argument('-hidden_size', type=int, default=768)
    group.add_argument('-num_heads', type=int, default=8)
    group.add_argument('-num_layers', type=int, default=1)
    group.add_argument('-dropout', type=float, default=0.1)
    group.add_argument('-fusion', type=str, default='add', choices=['cat', 'add'])

    group.add_argument('-seed', type=int, default=42)    
    group.add_argument('-num_epochs', type=int, default=10)
    group.add_argument('-end_step', type=int, default=3,
                       help='Dừng sớm (Early stopping) nếu sau 3 epoch không cải thiện')

def add_md_help_argument(parser):
    """ Thêm tùy chọn in giúp đỡ (Có thể giữ hoặc bỏ) """
    parser.add_argument('-md', action='store_true', help='In hướng dẫn.')

