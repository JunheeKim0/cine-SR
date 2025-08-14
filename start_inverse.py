#!/usr/bin/env python
# start_inverse.py

import os
import sys

def main():
    # 실행할 커맨드 구성
    cmd = ["torchrun", "--nproc_per_node=4", "LA_condition_all_pred_after.py"] + sys.argv[1:]
    # 현재 프로세스를 완전히 대체하여 torchrun 커맨드를 실행
    os.execvp("torchrun", cmd)

if __name__ == "__main__":
    main()
