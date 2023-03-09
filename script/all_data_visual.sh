#!/bin/bash
cd /home/user304/users/jiwon/graduation_paper/utils/
nohup python all_data_to_fifty.py >> ../shell/log/$(date +%m%d_%H%M).log &