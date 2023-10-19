from tree_sitter import Language, Parser

PY_LANGUAGE = Language('build/my-languages.so', 'python')


def remove_comments(program: str):
	parser = Parser()
	parser.set_language(PY_LANGUAGE)
	tree = parser.parse(bytes(program, "utf8"))
	
	# Get each instruction
	instructions = []
	for node in tree.root_node.children:
		if node.type == 'comment':
			continue
		instruction = program[node.start_byte:node.end_byte]
		instructions.append(instruction)
		
	return "\n".join(instructions)


def remove_empty_lines(program: str):
	lines = program.split('\n')
	lines = [line for line in lines if line.strip() != '']
	return "\n".join(lines)


def remove_non_ascii(program: str):
	return program.encode('ascii', 'ignore').decode()


def get_imported_lib(program: str):
	
	# Clean the program
	program = remove_empty_lines(program)
	program = remove_non_ascii(program)
	program = remove_comments(program)
	
	parser = Parser()
	parser.set_language(PY_LANGUAGE)
	tree = parser.parse(bytes(program, "utf8"))
	
	# From the tree, get imported libraries
	imported_libraries = []
	for node in tree.root_node.children:
		
		if node.type == 'import_statement':
			# Get all the imported libraries
			for child in node.children:
				if child.type == 'aliased_import':
					for grandchild in child.children:
						if grandchild.type == 'dotted_name':  # The other child is the alias, we keep the library name
							imported_libraries.append(program[grandchild.start_byte:grandchild.end_byte])
				
				elif child.type == 'dotted_name':
					imported_libraries.append(program[child.start_byte:child.end_byte])
		
		elif node.type == 'import_from_statement':
			# Set the library name as module_name.dotted_name
			module_name = program[node.children[1].start_byte:node.children[1].end_byte]
			for child in node.children[3:]:  # Skil from, <module_name>, import
				if child.type == 'aliased_import':
					for grandchild in child.children:
						if grandchild.type == 'dotted_name':  # The other child is the alias, we keep the library name
							imported_libraries.append(module_name + '.' + program[child.start_byte:child.end_byte])
				
				elif child.type == 'dotted_name':
					imported_libraries.append(module_name + '.' + program[child.start_byte:child.end_byte])
					
	return imported_libraries
			

	

