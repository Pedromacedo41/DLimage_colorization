import argparse


'''
Script for making a colorful video from a legacy black and white one
'''

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default="none", type=str)
    parser.add_argument('--input_file', default="none", type=str)
    parser.add_argument('--output_file', default="none", type=str)

    args = parser.parse_args()
    return args


def main(args):
    print(args)
    pass

if __name__ == '__main__': 
    args = parse_args()
    main(args)