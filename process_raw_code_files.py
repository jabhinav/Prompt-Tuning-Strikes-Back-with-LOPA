import os
from tqdm import tqdm

# Read python files in directory
_dir = './sample_codes'
files = os.listdir(_dir)
files = [f for f in files if f.endswith('.py')]

# Create new directory to store processed files
new_dir = './sample_codes_processed'
if not os.path.exists(new_dir):
	os.mkdir(new_dir)

# Read each file and remove comments
for f in tqdm(files, desc='Processing files', total=len(files)):
	with open(os.path.join(_dir, f), 'r') as file:
		lines = file.readlines()
		
		# Remove comments starting with '#'
		lines = [l for l in lines if not l.startswith('#')]
		
		# Remove inline comments
		lines = [l.split('#')[0] for l in lines]
		
		# Remove empty lines
		lines = [l for l in lines if l.strip() != '']
		
		# Remove trailing whitespace
		lines = [l.rstrip() for l in lines]
		
		# Remove Block comments
		for i in range(len(lines)):
			if lines[i].startswith('"""'):
				lines[i] = ''
				i += 1
				while not lines[i].endswith('"""'):
					lines[i] = ''
					i += 1
				lines[i] = ''
	
	# Re-write file
	with open(os.path.join(new_dir, f), 'w') as file:
		file.writelines('\n'.join(lines))
		