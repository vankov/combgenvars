from __future__ import print_function

import sys

from data import TrainSet, CombinatorialGeneralizationTestSet

tasks = [0]
VARS_tasks = [0, 1, 2, 3, 4]

print("Generating train set...", end = "")
sys.stdout.flush()
TrainSet.generate("data/train", tasks, VARS_tasks = VARS_tasks)
print("Done.")

print("Generating test set...", end = "")
sys.stdout.flush()
CombinatorialGeneralizationTestSet.generate(
            "data/test",
            tasks = tasks,
            VARS_tasks = VARS_tasks
        )
print("Done.")