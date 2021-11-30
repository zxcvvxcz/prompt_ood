from datasets import load_dataset, load_metric
import pdb
import os

def main():
    custom_dataset = load_dataset('json', os.path.join('data', 'clinc150', 'data_full.json'))



if __name__ == '__main__':
    main()