nums = [1,2,-3,4,-5]
genexp = (x>0 for x in nums)
for x in genexp:
    print(x)

if (x in nums
    for x in genexp):
    print(x)

if (x in genexp
    for x in genexp):
    print(x)

any(x in nums 
for x in genexp)
print(x)  

y = [False, False, False, True]  
any(y)
print(any(y))

for x in genexp:
    print(x)

# any in for in: test if there are elements overlapped! 
any(ttt in [1,2,3,4,5] for ttt in [-1,-2,-3,-4,-5])
any(ttt in [1,2,3,4,5] for ttt in [-1,-2,-3,-4,5])

sum = 0
for xt in [1,2,3,4,5]:
    sum += xt
    print(sum)
