def get_n_none(n):
    return [None] * n

def lru_get_none(n, ca={}):
    ca[n] = n
    print(ca)



if __name__ == '__main__':
    lru_get_none(3)
    lru_get_none(4)
