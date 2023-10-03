import os
import json
from typing import List, Dict

path = '../ec/data/list_tasks.json'
with open(path, 'r') as f:
	data: List[dict] = json.load(f)
	
# Print random task. Keys :=
# 	Type = {"input": "list-of-int", "output": "list-of-int"},
# 	Name = NL Desc.,
# 	Examples = {'i': [], 'o': []}
import random
print("Number of tasks: ", len(data))
print(json.dumps(random.choice(data), indent=4))
	
path = '../ec/data/list_tasks2.json'
with open(path, 'r') as f:
	data2 = json.load(f)
print("\n\nNumber of tasks: ", len(data2))
print(json.dumps(random.choice(data2), indent=4))
