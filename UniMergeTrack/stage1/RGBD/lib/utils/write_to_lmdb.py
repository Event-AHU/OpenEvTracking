import lmdb
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='generate lmdb')
# parser.add_argument('--base_dir', type=str, required=True, help='data directory.')
# parser.add_argument('--result_file', type=str, required=True, help="result file.")
args = parser.parse_args()
# base_dir = args.base_dir
# result_file = args.result_file

# base_dir = " data/got10k/"  # replace it with your own path
# result_file = " data/got10k.namelist"
# base_dir = " data/lasot/"  # replace it with your own path
# result_file = " data/lasot.namelist"
# base_dir = " data/vid/"  # replace it with your own path
# result_file = " data/vid.namelist"
# base_dir = " data/coco/"  # replace it with your own path
# result_file = " data/coco.namelist"
base_dir = ""  # replace it with your own path
result_file = ""

namelist = [x.strip() for x in open(result_file).readlines()]
print('number:', len(namelist))

if base_dir.endswith("/"):
    lmdb_fname = base_dir[:-1] + '_lmdb'
else:
    lmdb_fname = base_dir + '_lmdb'
env = lmdb.open(lmdb_fname, map_size=1024 ** 4)
txn = env.begin(write=True)

for i, t in enumerate(tqdm(namelist)):
    if i % 100000 == 0:
        txn.commit()
        txn = env.begin(write=True)
    with open(os.path.join(base_dir, t), 'rb') as fin:
        txn.put(key=t.encode(), value=fin.read())

txn.commit()
env.close()
