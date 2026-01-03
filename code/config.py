"""配置参数"""
import argparse

def get_config():
    p = argparse.ArgumentParser(description='BadNet Demo')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--poison_rate', type=float, default=0.1)
    p.add_argument('--target_label', type=int, default=0)
    p.add_argument('--trigger_size', type=int, default=5)
    p.add_argument('--data_path', default='./data')
    p.add_argument('--save_path', default='./models')
    return p.parse_args()
