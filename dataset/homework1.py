import math

def quadratic(a,b,c):

    x1 = ((-b) + math.sqrt((b * b) - 4 * a * c)) / 2 * a
    x2 = ((-b) - math.sqrt((b * b) - 4 * a * c)) / 2 * a
    print("方程 %0.1fx^2 + %0.1fx + %0.1f = 0 的解为：%0.2f %0.2f"%(a,b,c,x1,x2))
    return x1,x2

if __name__ == '__main__':

    x1,x2 = quadratic(1,1,-6)