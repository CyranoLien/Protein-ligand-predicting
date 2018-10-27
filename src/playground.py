
output=[0.001,-0.001,0.001]


edit_type = lambda x: 1 if x>0 else -1

output[1] = edit_type(output[1])
print(output)

