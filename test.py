import copy
a = [1,2,3]
b = copy.copy(a)
a.append(4)
print(b)