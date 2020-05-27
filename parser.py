import argparse

parser = argparse.ArgumentParser(description='Dynamic Pong RL')
parser.add_argument('--width', default=160, type=int, 
                    help='canvas width (default: 160)')
parser.add_argument('--height', default=160, type=int,
                    help='canvas height (default: 160)')
parser.add_argument('--ball', default=3.0, type=float,
                    help='ball speed (default: 3.0)')
parser.add_argument('--snell', default=3.0, type=float,
                    help='snell speed (default: 3.0)')
parser.add_argument('--ps','--paddle-speed', default=3.0, type=float,
                    help='paddle speed (default: 3.0)')
parser.add_argument('--pl','--paddle-length', default=45, type=int,
                    help='paddle length (default: 45)')
parser.add_argument('--lr','--learning-rate', default=1e-4, type=float,
                    help='learning rate (default: 1e-4)')

args = parser.parse_args()
parser.print_help()
