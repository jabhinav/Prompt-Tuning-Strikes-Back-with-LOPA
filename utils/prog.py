from typing import Tuple

import parso

from tree_sitter import Language, Parser




def remove_comments(program: str):
	PY_LANGUAGE = Language('build/my-languages.so', 'python')
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
	PY_LANGUAGE = Language('build/my-languages.so', 'python')

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


def identify_stmt_to_validate(program: str) -> Tuple[str, str]:
	lines = program.split('\n')
	
	# Remove empty lines
	lines = [line for line in lines if line.strip() != '']
	
	if len(lines) == 0:
		return None, None
	
	elif len(lines) == 1:
		return '', lines[0]
	
	else:
		running_indent = len(lines[0]) - len(lines[0].lstrip())
		valid_stmts = []
		last_found_return = -1
		for i, line in enumerate(lines):
			
			current_indent = len(line) - len(line.lstrip())
			
			if current_indent < running_indent:
				# Reached a stmt beyond the current block
				break
			
			elif current_indent > running_indent:
				# Found a stmt within the current block. Keep it
				
				# If it is a return stmt, keep it and clean it up
				if line.strip().startswith('return'):
					last_found_return = i
					indent_here = line[:len(line) - len(line.lstrip())]
					# Clean up the return statement
					eos = find_end_of_first_valid_python_expression(line.strip())
					cleaned_line = indent_here + line.strip()[:eos]
					valid_stmts.append(cleaned_line)
				
				else:
					valid_stmts.append(line)
			
			else:
				# Found a stmt at the same level as the current block
				# If it is not a return stmt, keep it
				if not line.strip().startswith('return'):
					# Found a return stmt within the current block. Keep it
					valid_stmts.append(line)
					
				else:
					last_found_return = i
					# For all lines below the return stmt including it, merge and return
					if line.endswith('\\'):
						valid_stmt = line[:-1].rstrip()
					else:
						valid_stmt = line.rstrip()
					
					if i + 1 < len(lines):
						for j in range(i + 1, len(lines)):
							# Include only if the next line is indented at the higher level than the return stmt
							if len(lines[j]) - len(lines[j].lstrip()) > running_indent:
								valid_stmt += ' ' + lines[j].strip()
							else:
								break
					valid_stmts.append(valid_stmt)
					break
				
		
		if len(valid_stmts) == 0:
			return '', program
		
		# Check if we finished looping -> This can happen
		# 1) If program has no return statement
		# 2) If all return statements are at the higher indent level as the current block: if: rtn1, else: rtn3
		if i == len(lines) - 1:
			if last_found_return > -1:
				correct_prefix = '\n'.join(valid_stmts[:last_found_return])
				
				# check if the return statement is split across multiple lines
				if last_found_return < len(lines) - 1:
					valid_stmt = valid_stmts[last_found_return]
					if valid_stmt.endswith('\\'):
						valid_stmt = valid_stmt[:-1].rstrip()
					else:
						valid_stmt = valid_stmt.rstrip()
						
					for j in range(last_found_return + 1, len(lines)):
						# Include only if the next line is indented at the higher level than the return stmt
						if len(lines[j]) - len(lines[j].lstrip()) > running_indent:
							valid_stmt += ' ' + lines[j].strip()
						else:
							break
					return correct_prefix, valid_stmt
				
				return correct_prefix, valid_stmts[last_found_return]
		
		correct_prefix = '\n'.join(valid_stmts[:-1])
		return correct_prefix, valid_stmts[-1]


def find_end_of_first_valid_python_expression(input_str: str) -> int:
	"""
	Find the end of a valid python expression in the input string
	Assumes there is only one expression in the input string
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
	prefix, final_valid_stmt = identify_stmt_to_validate(program)
	
	if final_valid_stmt is None:
		return program
	
	# # Keep the leading whitespaces is any
	indentation = final_valid_stmt[:len(final_valid_stmt) - len(final_valid_stmt.lstrip())]
	
	# Remove non-ascii characters from the return statement
	# cleaned_final_valid_stmt = remove_non_ascii(final_valid_stmt.strip())
	cleaned_final_valid_stmt = final_valid_stmt.strip()
	
	if not cleaned_final_valid_stmt.startswith('return') and 'return' in cleaned_final_valid_stmt:
		# Identify where the return statement starts
		start = cleaned_final_valid_stmt.index('return')
		_prefix = cleaned_final_valid_stmt[:start]
		_suffix = cleaned_final_valid_stmt[start:]
		end_of_expression = find_end_of_first_valid_python_expression(_suffix)
		if end_of_expression != -1:
			_suffix = _suffix[:end_of_expression]
		cleaned_final_valid_stmt = _prefix + _suffix
	else:

		end_of_expression = find_end_of_first_valid_python_expression(cleaned_final_valid_stmt)
		if end_of_expression != -1:
			cleaned_final_valid_stmt = cleaned_final_valid_stmt[:end_of_expression]
	
	# Reconstruct the program
	updated_program = prefix + '\n' + indentation + cleaned_final_valid_stmt
	
	return updated_program


if __name__ == '__main__':
	# Debug
	x = "    \n    if V == 0:\n        return 0\n    if m <= 0 or coins == []:\n        return float(\"inf\")\n    if coins[0] > V:\n        return min_coins(coins[1:], m, V)\n    else:\n        return min(min_coins(coins[1:], m, V - coins[0]), \n                  min_coins(coins, m-1, V) + 1) "
	print("Before:")
	print(x)
	print("\nAfter:")
	print(cleanup_return_statement(x))