import os

pwd = os.path.realpath(__file__)

name = os.path.dirname(pwd)
pname = os.path.dirname(name)
ppname = os.path.dirname(pname)
print(f"name:{name}")
print(f"pname:{pname}")
print(f"pname:{ppname}")