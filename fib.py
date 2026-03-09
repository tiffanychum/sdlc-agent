# Print Fibonacci sequence up to n terms
n = 10
f1, f2 = 0, 1
count = 0
while count < n:
    print(f1)
    f1, f2 = f2, f1 + f2
    count += 1
