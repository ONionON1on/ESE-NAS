# from Train import data_prepare
from Model_sampling import *

if __name__ == '__main__':

    print("process1")
    Ratings, LinkRatings, P, ct_up, ct_down, ct_all, link_ct_up, link_ct_down, link_ct_all = init_all(2)
    train_loader,test_loader=data_prepare()
    model_count=130
    x_data=[]
    y_data=[]
    model_list=[]
    for i in range (7,8):
        start=10
        #if i==6:
            # start=10
        for j in range(start,30):
            g = Random_Sampling(i, P)
            model_count += 1
            print("Process 1: Model No.", model_count, ":", g)
            model_list.append(g)
            model=cfnet(3,g,[1,2,2])
            time_per_epoch,acclist,best_acc = train_info(model,train_loader,test_loader,10,1)
            x_item = []
            x_item.append(time_per_epoch)
            for a in acclist:
                x_item.append(a)
            x_data.append(x_item)
            y_data.append(best_acc)

            #print("Process 1: Model No.",model_count)
            if model_count % 5 == 0:
                np.savez('/data/qiuzz/projectResearch/Results/Data1.npz',
                         x_data=x_data, y_data=y_data, models=model_list)
                print("10 more model information saved of Process 1")


    np.savez('/data/qiuzz/projectResearch/Results/Data1.npz',
             x_data=x_data,y_data=y_data,models=model_list)

