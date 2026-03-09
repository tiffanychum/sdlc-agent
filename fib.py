#!/usr/bin/env python3

def fibonacci(n):
    a, b = 0, 1
    seq = []
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq

if __name__ == "__main__":
    terms = 10
    print(" ".join(str(x) for x in fibonacci(terms)))
