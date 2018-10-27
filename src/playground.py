
output=[]


for i in range(5):
    for j in range(5):
        print(i, j)
        output.append(1 if i==j else 0)

print(output)