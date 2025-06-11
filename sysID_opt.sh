#!/usr/bin/bash

python3.11 toddlerbot/tools/sysID_opt.py --robot sysID_SM40BL  \
--data-folder run_policy_log/sysID_SM40BL_sysID_fixed_real_world_20250612_000449 \
--n-iters 1
