i = 3
j = 3
Tk = 0


for x in range(8):
    seq = 2**(i-j+1)
    active_range = seq // 2
    print(x,x % seq < active_range)
    
