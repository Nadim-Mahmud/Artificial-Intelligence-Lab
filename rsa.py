import math
import random


def coprimefind(phi):

    co = random.randint(1, phi-1)
    while math.gcd(co, phi) != 1:
        co = random.randint(1, phi-1)
    return co

def modInverse(a, m):
    for x in range(1, m):
        if (((a%m) * (x%m)) % m == 1):
            return x
    return -1


x = 61
y = 53

# calculating n and totient value phi
n = x*y
phi = (x-1)*(y-1)

e = coprimefind(phi)
d = modInverse(e, phi)


# encryption

message = 123

enc = pow(message, e)%n

print(enc)


# decrytion

dec = pow(enc, d) %n

print(dec)
