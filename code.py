a,b = map(int,input().split())
c = list(range(1,a+1))

for i in range(b):
    e,f = map(int,input().split())
    print(c[:e-1],c[e-1:f][::-1],c[f:])
print(*c)
