import os

def grade3_multiply(a:str, b:str)->int:
    intermediate = []
    int_a = int(a)
    lb = list(b)
    lb.reverse()
    for i, num in enumerate(lb):
        intermediate.append(int(num) * int_a * 10**i)
    return sum(intermediate)

def _karatsuba(a:str, b:str)->int:
    pass

if __name__ == "__main__":
    pass
