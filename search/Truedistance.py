def mdistance(p1, p2):
    print p1,p2
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

mincost=99
def truedistance(v, L, post):
    for (i, point) in enumerate(L):
        print 'V :%r'%v
        print 'L :%r'%L
        c = v
        c += mdistance(point, post)
        print 'C:%r'%c
        global mincost
        if v+c < mincost:
            if len(L) >=3:
                print L[0:i] + L[i + 1:]
                c += truedistance( c, L[0:i] + L[i + 1:], point)
            else :
                c+=mdistance(L[0],L[1])
                print c
                if c<mincost:
                    mincost=c
        else:
            pass
    return mincost

