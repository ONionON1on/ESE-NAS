from random import *
from copy import deepcopy
import numpy as np
# from numpy import *
from decode import *
from math import log

def isConnected(g):

    for i in range(1,len(g)):
        isCon=1
        for j in range(len(g)):
            if i in g[j][1]:
                isCon*=0
        if isCon!=0:
            return False

    return True



def randindex():
    return uniform(0,0.999999999999)


def get_Size(g):
    return len(g)
#Mutate

def Mutate(G,P,LinkRatings,mode): #modelsizecontrol
    g=deepcopy(G)
    size=len(g)
    # max=8
    # new_g = deepcopy(g)
    new_type = [0,0,0]
    # new_link=[0,0]
    new_link = []
    pt = randindex()
    if pt <= 0.7:
        pm = randindex()
        pa = 1-size/6
        if size <= 4:
            pc = 1
        else:
            pc = -0.5*(size-6)
        # Mutate the nodes
        if pm <= pa:
            print("A new node has been added!")
            prev_type, post_type, new_type, new_g=addNode(g, P, LinkRatings, mode)
            new_link.append([prev_type, new_type[0]])
            new_link.append([new_type[0], post_type])

        elif pa<pm<=pc:
            print("A node has been replaced!")
            new_type,new_g=changeNode(g,P)
        else:
            print("A node has been removed!")
            new_g=delNode(g)

    # mutate the links
    else:
        ifaddlink=randindex()

        sparsity=0.3
        link_count=0
        for i in range(size):
            link_count+=len(g[i][1])

        upbound=int(sparsity*(size+1)*(size+1))+1
        pl=-(link_count-upbound)/upbound

        # ifremlink=randindex()
        # if ifaddlink<=pl:
        if link_count<=upbound:
            flag,prev_type,post_type,new_g=addLink(g,LinkRatings,mode)
            if flag==1:
                print("New connection between Nodes!")
                new_link.append([prev_type,post_type])
        #elif ifaddlink>pl:
        else:
            new_g=delLink(g)
        # print("A connection has been REMOVED!")
    #TODO: return the Mutation Change.
    return new_type,new_link, new_g


#Ptype[op,kernel_size/reduction_rate,expansion_rate]

def pickOp(Ptype):
    TypeVec=[]
    t1=np.random.choice(7,1,Ptype[0])[0]+1
    TypeVec.append(t1)
    t3=0
    if t1<3 or t1==6:
        t2=2*np.random.choice(4,1,Ptype[t1])[0]+1 #Conv kernel_size 1,3,5,7
    elif t1==3:
        t2 = 2 * np.random.choice(4, 1, Ptype[t1][0])[0] + 1 #Conv kernel_size 1,3,5,7
        t3 = (2 ** (np.random.choice(3, 1, Ptype[t1][1])[0] + 1))  # expansion_rate 2,4,8
    elif 4<=t1<=5:
        t2 = 2 * np.random.choice(2, 1, Ptype[t1])[0] + 1  #pooling kernel_size 3,5
    else:
        t2=2*(2**(np.random.choice(4,1,Ptype[t1])[0]+1)) # reduction_rate 4,8,16,32
    TypeVec.append(t2)
    '''
    if t1==3:
        t3=(2**(np.random.choice(3,1,Ptype[4])[0]+1) # expansion_rate 2,4,8
    else:
        t3=0
    '''
    TypeVec.append(t3)
    return TypeVec


#initial Distribution
def addNode(g,P,linkRatings,mode):
    new_g=deepcopy(g)

    new_type=pickOp(P)
    new=[]
    new.append(new_type)

    s=topoSort(g)
    num=len(s)

    # prev=randint(0,num-1)
    prev,post=select_location(new_type,new_g,linkRatings,mode)
    prev_type = new_g[s[prev]][0][0]
    # print(prev,post)
    if prev==num-1:
        #print("1: prev=",s[prev],", post=out")
        new.append([100])
        new_g.append(new)
        new_g[s[prev]][1].remove(100)
        new_g[s[prev]][1].append(num)
        #prev_type=new_g[s[prev]][0][0]
        post_type=8
    else:
        # post=randint(prev+1,num)
        if post==num:
            new.append([100])
            #print("2: prev=", s[prev], ", post=out")
            post_type=8
        else:
            new.append([s[post]])
            #print("2: prev=", s[prev], ", post=", s[post])
            post_type=new_g[s[post]][0][0]
        new_g.append(new)
        new_g[s[prev]][1].append(num)
    #new_g[s[prev - 1]][1].append(num)
    return prev_type,post_type,new_type,new_g

def changeNode(g,P):
    new_g=deepcopy(g)
    newtype=pickOp(P)
    num = len(g)
    old=randint(1,num-1)
    new_g[old][0]=newtype
    return newtype,new_g

def delNode(g):
    new_g = deepcopy(g)
    delete=randint(1,len(g)-1)

    print("delete",delete)
    # prev->next = delete->next
    for i in range(len(g)):
        if delete in new_g[i][1]:
            new_g[i][1].remove(delete)
            for j in new_g[delete][1]:
                if j not in new_g[i][1]:
                    new_g[i][1].append(j)
    for i in range(len(g)):
        if len(new_g[i][1])!=0:
            for j in range(len(new_g[i][1])):
                if new_g[i][1][j]>delete and new_g[i][1][j] !=100:
                    new_g[i][1][j]-=1

    new_g.pop(delete)
    return new_g

# linkRatings: A 9*9 Matrix

def select_location(new_type,G,linkRatings,mode): # for ADDNODE:
    #prev=0
    #post=0
    # print(linkRatings)
    g=deepcopy(G)
    Rprev=[]
    Rpost=[]
    #
    s=topoSort(g)
    num=len(g)
    for i in range(num):
        #print(g[s[i]][0][0])
        #t=g[s[i]][0][0]
        ri=linkRatings[g[s[i]][0][0]][new_type[0]]
        Rprev.append(ri)
    #print(Rprev)
    #print()
    Pprev=getNewDistribution(Rprev,mode)
    #print(Pprev)
    prev=np.random.choice(num,1,Pprev)[0]
    if prev==num-1:
        post=num
        return prev,post

    for i in range(prev+1,num):
        ri=linkRatings[new_type[0]][g[s[i]][0][0]]
        Rpost.append(ri)
    Rpost.append(linkRatings[new_type[0]][8])
    Ppost=getNewDistribution(Rpost,mode)
    post=prev+1+np.random.choice(num-prev,1,Ppost)[0]
    #if post==num:
        #post=100
    return prev,post

def select_link(g,LinkRatings,mode):
    # prev=0
    #post=0

    s=topoSort(g)
    num=len(g)
    typeList=[]
    for v in s:
        typeList.append(g[v][0][0])

    Rlink=[]
    links=[]
    for i in range(num):
        for j in range(i+1,num+1):
            if j!=num:
                #links.append([i,j])
                Rlink.append(LinkRatings[typeList[i]][typeList[j]])
            else:
                Rlink.append(LinkRatings[typeList[i]][8]) #output Node
            links.append([i,j])

    Plinks=getNewDistribution(Rlink,mode)
    linkindex=np.random.choice(len(links),1,Plinks)[0]
    prev=links[linkindex][0]
    post=links[linkindex][1]
    if post==num:
        post=100

    return prev,post

def addLink(g,LinkRatings,mode):# link:prev_type,post_type
    # type: # 0=input 1-7=OpType 8=output
    new_g = deepcopy(g)
    s = topoSort(g)
    #print(s)
    num = len(s)
    #new_link=[]
    # TODO: select prev and post type_num
    # Need another func: def select_loc(new_type,G,LinkRatings)
    #prev = randint(0, num - 2)
    #new_link.append(new_g[s[prev]][0][0])
    #post = randint(prev + 1, num)
    prev,post=select_link(g,LinkRatings,mode)
    prev_type=new_g[s[prev]][0][0]
    flag=0
    post_type=0
    if post==100:
        # post=100
        if post not in new_g[s[prev]][1]:
            new_g[s[prev]][1].append(post)
            flag=1
            post_type=8
            #return flag,prev_type,post_type,new_g
    # print(s[prev],s[post])
    elif post !=100 and s[post] not in new_g[s[prev]][1]:
        new_g[s[prev]][1].append(s[post])
        flag=1
        post_type=new_g[s[post]][0][0]

    return flag,prev_type,post_type,new_g

def delLink(g):
    new_g = deepcopy(g)
    s = topoSort(g)
    # print(s)
    num = len(s)
    prev = randint(0, num - 2)
    post_num=len(g[prev][1])
    if post_num==1:
        if prev!=0:
            new_g[prev][1][0]=100
        else:
            return new_g
    else:
        post = randint(0, post_num-1)
        new_g[prev][1].pop(post)

    if isConnected(new_g)==False:
        new_g=deepcopy(g)
    else:
        print("A link has been deleted!")
    return new_g

def update_ct(nt,ct,ct_all):
    ct[0][nt[0]-1] += 1
    ct_all[0][nt[0]-1] += 1
    # op++
    if nt[0]<3 or nt[0]==6: # ct[type][param1]
        ct[nt[0]][(nt[1]-1)/2] += 1
        ct_all[nt[0]][(nt[1]-1)/2] += 1
    elif nt[0]==3: #ct[3][0][param1]/ct[3][1][param2]
        ct[nt[0]][0][(nt[1]-1)/2] += 1
        ct_all[nt[0]][0][(nt[1]-1)/2] += 1
        ct[nt[0]][1][log(nt[2],2)-1] += 1
        ct_all[nt[0]][1][log(nt[2],2)-1] += 1
    elif 4<=nt[0]<=5:
        ct[nt[0]][0][(nt[1] - 3) / 2] += 1
        ct_all[nt[0]][0][(nt[1] - 3) / 2] += 1
    elif nt[0]==7:
        ct[nt[0]][1][log(nt[1], 2) - 1] += 1
        ct_all[nt[0]][1][log(nt[1], 2) - 1] += 1
    return



def update_rt(dacc,rating,ct,ctall, round):
    #print(dacc,rating,ct,ctall,round)
    beta=1-1/np.exp(round)
    new_rating=rating*(1+dacc*(ct/ctall)*beta)
    return new_rating

def getNewDistribution(Ratings,mode):
    if mode==1: # Softmax
        P=np.exp(Ratings) / np.sum(np.exp(Ratings), axis=0)
    elif mode==2: # Average
        P=Ratings/np.sum(Ratings)
    else:
        print("Invalid MODE selection!")
    return list(P)



def update_P(Ratings,mode):
    P= [getNewDistribution(Ratings[0], mode), getNewDistribution(Ratings[1], mode),
         getNewDistribution(Ratings[2], mode),
         [getNewDistribution(Ratings[3][0], mode), getNewDistribution(Ratings[3][1], mode)],
         getNewDistribution(Ratings[4], mode), getNewDistribution(Ratings[5], mode),
         getNewDistribution(Ratings[6], mode), getNewDistribution(Ratings[7], mode)]
    return P


def update_links(dacc,prev_type,post_type,LinkRatings,link_ct,link_ct_all,round):

    link_ct[prev_type][post_type]+=1
    link_ct_all[prev_type][post_type]+=1

    LinkRatings[prev_type][post_type]=update_rt(dacc,
                                                LinkRatings[prev_type][post_type],
                                                link_ct[prev_type][post_type],
                                                link_ct_all[prev_type][post_type],round)


    return

def update_all(dacc,new_type,Ratings,ct,ct_all,round):
    ct[0][new_type[0] - 1] += 1
    ct_all[0][new_type[0] - 1] += 1
    Ratings[0][new_type[0] - 1] = update_rt(dacc,
                                            Ratings[0][new_type[0] - 1],
                                            ct[0][new_type[0] - 1],
                                            ct_all[0][new_type[0] - 1], round)
    # op++
    if new_type[0] < 3 or new_type[0] == 6:  # ct[type][param1]
        ct[new_type[0]][(new_type[1] - 1) // 2] += 1
        ct_all[new_type[0]][(new_type[1] - 1) // 2] += 1
        Ratings[new_type[0]][(new_type[1] - 1) // 2] = update_rt(dacc,
                                                                Ratings[new_type[0]][(new_type[1] - 1) // 2],
                                                                ct[new_type[0]][(new_type[1] - 1) // 2],
                                                                ct_all[new_type[0]][(new_type[1] - 1) // 2], round)
    elif new_type[0] == 3:  # ct[3][0][param1]/ct[3][1][param2]
        ct[new_type[0]][0][(new_type[1] - 1) // 2] += 1
        ct_all[new_type[0]][0][(new_type[1] - 1) // 2] += 1
        Ratings[new_type[0]][0][(new_type[1] - 1) // 2] = update_rt(dacc,
                                                                   Ratings[new_type[0]][0][(new_type[1] - 1) // 2],
                                                                   ct[new_type[0]][0][(new_type[1] - 1) // 2],
                                                                   ct_all[new_type[0]][0][(new_type[1] - 1) // 2], round)
        ct[new_type[0]][1][int(log(new_type[2], 2) - 1)] += 1
        ct_all[new_type[0]][1][int(log(new_type[2], 2) - 1)] += 1
        Ratings[new_type[0]][1][int(log(new_type[2], 2) - 1)] = update_rt(dacc,
                                                                     Ratings[new_type[0]][1][int(log(new_type[2], 2) - 1)],
                                                                     ct[new_type[0]][1][int(log(new_type[2], 2) - 1)],
                                                                     ct_all[new_type[0]][1][int(log(new_type[2], 2) - 1)],
                                                                     round)
    elif 4 <= new_type[0] <= 5:
        ct[new_type[0]][(new_type[1] - 3) // 2] += 1
        ct_all[new_type[0]][(new_type[1] - 3) // 2] += 1
        Ratings[new_type[0]][(new_type[1] - 3) // 2] = update_rt(dacc,
                                                                Ratings[new_type[0]][(new_type[1] - 3) // 2],
                                                                ct[new_type[0]][(new_type[1] - 3) // 2],
                                                                ct_all[new_type[0]][(new_type[1] - 3) // 2], round)
    elif new_type[0] == 7:
        ct[new_type[0]][int(log(new_type[1], 4) - 1)] += 1
        ct_all[new_type[0]][int(log(new_type[1], 4) - 1)] += 1
        Ratings[new_type[0]][int(log(new_type[1], 4) - 1)] = update_rt(dacc,
                                                                  Ratings[new_type[0]][int(log(new_type[1], 4) - 1)],
                                                                  ct[new_type[0]][int(log(new_type[1], 4) - 1)],
                                                                  ct_all[new_type[0]][int(log(new_type[1], 4) - 1)], round)

    return