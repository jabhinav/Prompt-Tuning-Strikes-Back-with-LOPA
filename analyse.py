import json
from tqdm import tqdm
from collections import OrderedDict
from utils.prog import get_imported_lib

path_to_data = './data/APPS/train.jsonl'
data = []
with open(path_to_data, 'r') as f:
	for line in f:
		data.append(json.loads(line))
	
library_counts = OrderedDict()

# Convert to readable format
for idx, d in tqdm(enumerate(data), desc='Converting to readable format', total=len(data)):
	
	d['solutions'] = json.loads(d['solutions'])
	# d['input_output'] = json.loads(d['input_output']) if d['input_output'] is not None else None
	
	# print("\n# ################## {} ################## #".format(d["id"]))
	
	if isinstance(d['solutions'], list):
		for soln in d['solutions']:
			program = d['starter_code'] + soln
			imported_libs = get_imported_lib(program)
			# if len(imported_libs) > 0:
			# 	print("Imported Libraries: ", imported_libs)
			for lib in imported_libs:
				if lib not in library_counts:
					library_counts[lib] = 0
				library_counts[lib] += 1
	else:
		program = d['starter_code'] + d['solutions']
		imported_libs = get_imported_lib(program)
		# if len(imported_libs) > 0:
		# 	print("Imported Libraries: ", imported_libs)
		for lib in imported_libs:
			if lib not in library_counts:
				library_counts[lib] = 0
			library_counts[lib] += 1


library_counts = OrderedDict(sorted(library_counts.items(), key=lambda x: x[1], reverse=True))

# Print the counts
print("\n# ################## Library Counts ################## #")
for lib, count in library_counts.items():
	print(lib, ": ", count)
	
# Dump the counts
with open('library_counts.json', 'w') as f:
	json.dump(library_counts, f, indent=4)
