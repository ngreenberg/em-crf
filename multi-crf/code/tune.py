import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='exp1.py', type=str)
args = parser.parse_args(sys.argv[1:])

partition = 'titanx-long'
time = '06-10:00:00'
log_dir = './logs/%s' % args.filename[:-3]
user = 'ngreenberg'


slurm_cmd = 'srun --gres=gpu:1 --partition=%s --time=%s' % (partition, time)
base_cmd = 'python %s' % args.filename
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

max_total_jobs = 100
max_jobs_at_once = 2

hiddensize = [100, 200, 300]
dropout = [0.5, 0.75, 1.0]

all_params = [hiddensize, dropout]
names = ['hiddensize', 'dropout']

all_jobs = list(itertools.product(*all_params))
random.shuffle(all_jobs)
jobs_list = {}
for i, setting in enumerate(all_jobs[:max_total_jobs]):
    name_setting = {n: s for n, s in zip(names, setting)}
    # remove redundant settings
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n, str(s)) for n, s in name_setting.items()])
    jobs_list[log_str] = setting_str

print('Running %d jobs and writing logs to %s' % (len(jobs_list), log_dir))
for log_str, setting_str in jobs_list.items():
    full_cmd = '%s %s %s' % (slurm_cmd, base_cmd, setting_str)
    bash_cmd = '%s &> %s/%s &' % (full_cmd, log_dir, log_str)

    # only run max_jobs_at_once at once
    jobs = max_jobs_at_once
    while jobs >= max_jobs_at_once:
        jobs = int(subprocess.check_output('squeue  | grep %s | wc -l' % user, shell=True))
        sleep(1)
    print(bash_cmd)
    subprocess.call(bash_cmd, shell=True)

print('Done. Ran %d jobs.' % len(jobs_list))
