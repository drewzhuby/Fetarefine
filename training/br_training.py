"""Retired standalone BR entry; all stages now use YAML-bound train.py.

The previous implementation is retained in audits/retrain_repair/before/.
"""
def run_br(*args, **kwargs):
    raise RuntimeError("Use train.py --model m5_br --phase phase2; direct legacy BR entry is disabled")
