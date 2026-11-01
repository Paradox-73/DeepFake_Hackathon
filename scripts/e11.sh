#!/bin/bash
# Screen of methods not tried yet: new image views, pooling, older and newer backbones, then optimiser details and colour damage.
scripts/run_jobs.sh configs/jobs_e10c.txt 2 configs/exp.yaml
scripts/run_jobs.sh configs/jobs_e11c.txt 2 configs/exp.yaml
echo E11C_FINISHED
scripts/run_jobs.sh configs/jobs_e11a.txt 2 configs/exp.yaml
echo E11A_FINISHED
scripts/run_jobs.sh configs/jobs_e11d.txt 1 configs/exp.yaml
echo E11D_FINISHED
scripts/run_jobs.sh configs/jobs_e11b.txt 2 configs/exp.yaml
echo E11_DONE
