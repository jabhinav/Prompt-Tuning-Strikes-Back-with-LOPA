import parso

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


def identify_final_return_statement(program: str) -> str:
	"""
	Identify the line containing the final return statement in the program i.e. the return statement that ends the
	function block within which it is present
	
	> Uses indentation
	> Can't handle return statement spanning multiple lines. In such cases it simply truncates the return statement at the first line
	:param program:
	:return:
	"""
	
	lines = program.split('\n')

	# Remove empty lines
	lines = [line for line in lines if line.strip() != '']
	
	if len(lines) == 0:
		return None
	
	elif len(lines) == 1:
		return lines[0]
	
	else:
		running_indent = len(lines[0]) - len(lines[0].lstrip())
		
		return_stmts = []
		for line in lines:
			current_indent = len(line) - len(line.lstrip())
			if line.strip().startswith('return'):
				if current_indent < running_indent:
					# Reached a return stmt beyond the current block
					break
				elif current_indent == running_indent:
					# Found the return stmt ending the current block
					return_stmts.append(line)
					break
				else:
					# Found a return stmt within the current block. Keep it
					return_stmts.append(line)
		
		if len(return_stmts) == 0:
			return program
		
		return return_stmts[-1]
	
	
def find_end_of_expression(input_str: str) -> int:
	"""
	Find the end of a valid python expression in the input string
	:param input_str:
	:return:
	"""
	try:
		tree = parso.parse(input_str)
		expr = tree.children[0]
		return expr.end_pos[1]
	except (parso.parser.ParserSyntaxError, IndexError):
		return -1


def cleanup_return_statement(program: str) -> str:
	"""
	Locates the final return statement in the program and cleans it up.
	Useful when LLM generates gibberish after the return statement
	:return:
	"""
	return_statement = identify_final_return_statement(program)
	
	if return_statement is None:
		return program
	
	# # Keep the leading whitespaces is any
	indentation = return_statement.split('return')[0]
	
	# Remove non-ascii characters from the return statement
	cleaned_return_statement = remove_non_ascii(return_statement.strip())
	cleaned_return_statement = cleaned_return_statement.strip()

	end_of_expression = find_end_of_expression(cleaned_return_statement)
	if end_of_expression != -1:
		cleaned_return_statement = cleaned_return_statement[:end_of_expression]
	
	# Find the last occurance of the return statement in the program and split the program at that point
	program = program.rsplit(return_statement, 1)[0]
	
	# Reconstruct the program
	updated_program = program + indentation + cleaned_return_statement
	
	return updated_program
	
	

