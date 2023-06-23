
from Train import *
#from torchsummary import summary
from ParamSaveLoad import *
from mutation import *

#local_path="/home/qiuzz/results"
#ssh_path="/data/qiuzz/projectResearch/results"

def SelectNextGen(Sons,sonPerf): # Parents=6
    sonnum=len(Sons)
    # Fathers=[]
    # fatherPerf=[]
    Ranks = [-1, -1, -1, -1, -1, -1]
    for i in range(sonnum):
        if Ranks[0] == -1 or (
                sonPerf[i] > sonPerf[Ranks[0]]): # and sonPerf[i][1] >= sonPerf[Ranks[0]][1]):
            Ranks[5] = Ranks[4]
            Ranks[4] = Ranks[3]
            Ranks[3] = Ranks[2]
            Ranks[2] = Ranks[1]
            Ranks[1] = Ranks[0]
            Ranks[0] = i
            continue
        if Ranks[1] == -1 or (
                sonPerf[i] > sonPerf[Ranks[1]]): # and sonPerf[i][1] >= sonPerf[Ranks[1]][1]):
            Ranks[5] = Ranks[4]
            Ranks[4] = Ranks[3]
            Ranks[3] = Ranks[2]
            Ranks[2] = Ranks[1]
            Ranks[1] = i
            continue
        if Ranks[2] == -1 or (
                sonPerf[i] > sonPerf[Ranks[2]]): # and sonPerf[i][1] >= sonPerf[Ranks[2]][1]):
            Ranks[5] = Ranks[4]
            Ranks[4] = Ranks[3]
            Ranks[3] = Ranks[2]
            Ranks[2] = i
            continue
        if Ranks[3] == -1 or (
                sonPerf[i] > sonPerf[Ranks[3]]): # and sonPerf[i][1] >= sonPerf[Ranks[3]][1]):
            Ranks[5] = Ranks[4]
            Ranks[4] = Ranks[3]
            Ranks[3] = i
            continue
        if Ranks[4] == -1 or (
                sonPerf[i] > sonPerf[Ranks[4]]): # and sonPerf[i][1] >= sonPerf[Ranks[3]][1]):
            Ranks[5] = Ranks[4]
            Ranks[4] = i
            continue
        if Ranks[5] == -1 or (
                sonPerf[i] > sonPerf[Ranks[5]]): # and sonPerf[i][1] >= sonPerf[Ranks[3]][1]):
            Ranks[5] = i
            continue
    return Ranks


def init_all(mode):
    Ratings = [[1.0] * 7, [1.0] * 4, [1.0] * 4, [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [1.0] * 2, [1.0] * 2, [1.0] * 4, [1.0] * 4]
    LinkRatings=[[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9,[1.0]*9]
    for i in range(9):
        LinkRatings[i][0]=0.0
        LinkRatings[8][i]=0.0
    # Ratings[0][0]=0
    P = [getNewDistribution(Ratings[0], mode), getNewDistribution(Ratings[1], mode),
         getNewDistribution(Ratings[2], mode),
         [getNewDistribution(Ratings[3][0], mode), getNewDistribution(Ratings[3][1], mode)],
         getNewDistribution(Ratings[4], mode), getNewDistribution(Ratings[5], mode),
         getNewDistribution(Ratings[6], mode), getNewDistribution(Ratings[7], mode)]

    ct_up = [[0] * 7, [0] * 4, [0] * 4, [[0, 0, 0, 0], [0, 0, 0]], [0] * 2, [0] * 2, [0] * 4, [0] * 4]
    ct_down = [[0] * 7, [0] * 4, [0] * 4, [[0, 0, 0, 0], [0, 0, 0]], [0] * 2, [0] * 2, [0] * 4, [0] * 4]
    ct_all = [[0] * 7, [0] * 4, [0] * 4, [[0, 0, 0, 0], [0, 0, 0]], [0] * 2, [0] * 2, [0] * 4, [0] * 4]

    link_ct_up=[[0]*9,[0]*9,[0]*9,[0]*9,[0]*9,[0]*9,[0]*9,[0]*9,[0]*9]
    link_ct_down = [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]
    link_ct_all = [[0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9, [0] * 9]
    return Ratings,LinkRatings,P,ct_up,ct_down,ct_all,link_ct_up,link_ct_down,link_ct_all

def main(search_mode,update_mode,train_loader, test_loader):
    print("Search has started!")
    # P=[[0.2,0.2,0.2,0.1,0.1,0.1,0.1],[0.25,0.25,0.25,0.25],[0.5,0.5],[0.25,0.25,0.25,0.25],[0.33,0.33,0.34]]
    '''
    if path.isfile('/data/qiuzz/projectResearch/Fathers.npy'):
        Fathers=np.load('Fathers.npy',allow_pickle=True)
        Fathers=Fathers.tolist()
    else:
    '''

    Fathers=[
        [[[0, 0, 0], [1]], [[2, 3, 0], [100]]]
    ]
    fatherPerf=[]
    father_Dicts=[]
    best_acc = 0
    best_arch = None
    best_state_dict = None
    all_performance=[]
    # ======================================================
    # TODO: Ratings for all ops and parameters
    Ratings,LinkRatings, P, ct_up, ct_down, ct_all,link_ct_up,link_ct_down,link_ct_all=init_all(update_mode)
    # ======================================================
    print("start at", time.asctime(time.localtime(time.time())))
    startat=time.time()
    print("============Initialize the First Generation==============")
    # print("start at",time.asctime(time.localtime(time.time())))
    for father in Fathers:
        print("father:", father)
        currentModel=cfnet(3,father,[1,2,2])
        # currentPerf, weights=eval(currentModel,train_loader, test_loader)
        # currentPerf= eval(currentModel, train_loader, test_loader)
        currentPerf=eval_via_predictor(currentModel,train_loader,test_loader)
        weights = None
        # acc_estimated=log(currentPerf)/5+0.4
        # currentPerf=0.91
        print("Performance:", currentPerf)
        # print("Estimating the ACCURACY(Not specific):", acc_estimated)

        # sonPerf.append(currentPerf)
        fatherPerf.append(currentPerf)
        father_Dicts.append(weights)
        if currentPerf > best_acc:
            best_arch = father
            best_state_dict = weights
            # torch.save(currentModel,"/data/qiuzz/projectResearch/cnn/best_model_so_far.pth")
            best_acc=currentPerf
    Sons = []
    sonPerf = []
    son_Dicts=[]
    whoseSon=[]
    #Ranks=[-1,-1,-1,-1]
    UPPERBOUND=5
    #Keep tracks on the best result so far.
    all_perf=[]
    Performances=[]

    # ======================================================
    for round in range(1,32):
        print("===============round",round,"==============")
        count=1
        perf = []
        for i in range(len(Fathers)):
            # This MUTATION & EVALUATION block could be repeated a FEW times
            Sons.append(Fathers[i])
            sonPerf.append(fatherPerf[i])
            perf.append(fatherPerf[i])
            son_Dicts.append(father_Dicts[i])
            if round==1:
                num=6
            else:
                num=4
            for t in range(num):
                new_type,new_link,son=Mutate(Fathers[i],P,LinkRatings,update_mode) # should return mutation information
                Sons.append(son)
                print("current arch: No." + str(count) + " ", son)
                print("Evolve from Father No."+str(i))
                count+=1
                currentModel=cfnet(3,son,[1,2,2])
                
                currentPerf = eval_via_predictor(currentModel, train_loader, test_loader)
                perf.append(currentPerf)
               
                print("Performance:", currentPerf)
                
                sonPerf.append(currentPerf)

                # ======================================================
                # TODO: if better result comes up?
                if currentPerf>best_acc:
                    print("Better results comes up!")
                    best_acc=currentPerf
                    best_arch=son
                    best_state_dict=weights
                    # torch.save(currentModel,'/data/qiuzz/projectResearch/cnn/best_model_so_far.pth')
                    #print("Best model saved!")
                # ======================================================
                # TODO: update the distribution P

                if search_mode==1:
                    if currentPerf>=fatherPerf[i]:
                        dacc=0.1
                    else:
                        dacc=-0.1
                    if new_type[0]!=0 and dacc>=0:
                        #op update
                        update_all(dacc,new_type,Ratings,ct_up,ct_all,round)
                    if new_type[0]!=0 and dacc<0:
                        update_all(dacc, new_type, Ratings, ct_down, ct_all,round)
                    P = update_P(Ratings, update_mode)

                    if len(new_link)!=0:
                        for link in new_link:
                            if dacc>=0:
                                update_links(dacc,link[0],link[1],LinkRatings,link_ct_up,link_ct_all,round)
                            else:
                                update_links(dacc,link[0],link[1],LinkRatings,link_ct_down,link_ct_all,round)

                # ======================================================
                # No need for Random Search.
                whoseSon.append(i)
                # TODO:Save state_dict
                son_Dicts.append(Sons)
        all_perf.append(perf)
            # ======================================================

        # Fathers,fatherPerf=SelectNextGen(Sons,sonPerf)
        Ranks=SelectNextGen(Sons,sonPerf)
        Fathers=[]
        fatherPerf=[]
        print("Fathers for Next Round:")
        for i in range(6):
            Fathers.append(Sons[Ranks[i]])
            print(Sons[Ranks[i]])
            fatherPerf.append(sonPerf[Ranks[i]])
            print("Performance:",sonPerf[Ranks[i]])
            father_Dicts.append(son_Dicts[Ranks[i]])
        np.save('Fathers.npy',Fathers)
        Sons=[]
        sonPerf=[]
        Performances.append(fatherPerf[0])
        # TODO: SAVE what we need. Ratings
        np.savez('/data/qiuzz/projectResearch/cnn/Ratings0805.npz',Ops=Ratings,Links=LinkRatings)
        # np.save('/data/qiuzz/projectResearch/cnn/PerfProcess_ramdom220730_p1.npy',Performances)
        # np.save('/data/qiuzz/projectResearch/cnn/PerfProcess_ESENAS220805.npy', Performances)
        np.save('/data/qiuzz/projectResearch/cnn/PerfProcess_ESENAS220805.npy', all_perf)
        print("this round stop at",time.asctime(time.localtime(time.time())))
        print('the program has been running for',time.time()-startat)
        # ======================================================

    return best_acc,best_arch,best_state_dict





if __name__=='__main__':
    train_loader, test_loader = data_prepare()
    main(1, 1, train_loader, test_loader)
    print("ends at", time.asctime(time.localtime(time.time())))
   