"""Write a job file: every variant crossed with every fold.
Usage: python scripts/gen_jobs.py configs/jobs_e5.txt "0 1 2" "<base overrides>" name1="overrides" name2="overrides" """
import sys

out, folds, base, *variants = sys.argv[1:]
with open(out, "w", newline="\n") as f:
    for v in variants:
        name, _, over = v.partition("=")
        for k in folds.split():
            f.write(" ".join(("%s_f%s data.fold=%s %s %s" % (name, k, k, base, over)).split()) + "\n")
