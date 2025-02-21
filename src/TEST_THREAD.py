import sys
a = "hello"
print(sys.getrefcount(a))
b = a
c = a
print(sys.getrefcount(a))
c = 100000
print(sys.getrefcount(a))
del b
print(sys.getrefcount(a))

print("====")
print(sys.getrefcount(c))