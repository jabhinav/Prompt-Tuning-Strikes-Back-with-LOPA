# Python code to find sum
# of natural numbers upto
# n using recursion

# Returns sum of first
# n natural numbers
def recurSum(n):
	if n <= 1:
		return n
	return n + recurSum(n - 1)


# Driver code
n = 5
print(recurSum(n))