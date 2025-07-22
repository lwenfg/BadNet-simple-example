import argparse

def get_config():
    parser = argparse.ArgumentParser(description='MNIST Backdoor Attack Demo')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--trigger_label', type=int, default=0)
    parser.add_argument('--trigger_size', type=int, default=5)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--trigger_path', default='./triggers/trigger_white.png')
    parser.add_argument('--model_save_dir', default='./models')
    return parser.parse_args()