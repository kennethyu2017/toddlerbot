#!/usr/bin/bash

python3.11 toddlerbot/tools/sysID_opt.py --robot sysID_SM40BL  \
--data-folder run_policy_log/sysID_SM40BL_sysID_fixed_real_world_20250614_222232  \
--n-iters 200 \
--n-jobs 10  \
--early-stop 200
#--eval-only true


