# Print Fibonacci sequence up to 10 terms
n = 10
a, b = 0, 1
for i in range(n):
    print(a)
    a, b = b, a + b
