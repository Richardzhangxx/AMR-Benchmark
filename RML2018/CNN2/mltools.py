import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import csv
# Show loss curves
def show_history(history):
    plt.figure()
    plt.title('Training loss performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()
    plt.savefig('figure/total_loss.png')
    plt.close()
 

    plt.figure()
    plt.title('Training accuracy performance')
    plt.plot(history.epoch, history.history['acc'], label='train_acc')
    plt.plot(history.epoch, history.history['val_acc'], label='val_acc')
    plt.legend()    
    plt.savefig('figure/total_acc.png')
    plt.close()

    train_acc=history.history['acc']
    val_acc=history.history['val_acc']
    train_loss=history.history['loss']
    val_loss=history.history['val_loss']
    epoch=history.epoch
    np_train_acc=np.array(train_acc)
    np_val_acc=np.array(val_acc)
    np_train_loss=np.array(train_loss)
    np_val_loss=np.array(val_loss)
    np_epoch=np.array(epoch)
    np.savetxt('train_acc.txt',np_train_acc)
    np.savetxt('train_loss.txt',np_train_loss)
    np.savetxt('val_acc.txt',np_val_acc)
    np.savetxt('val_loss.txt',np_val_loss)

def plot_lstm2layer_output(a,modulation_type=None,save_filename=None):
    plt.figure(figsize=(4,3),dpi=600)
    plt.plot(range(128),a[0],label=modulation_type)
    plt.legend()
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])
    plt.savefig(save_filename,dpi=600,bbox_inches ='tight')
    plt.tight_layout()
    plt.close()

def plot_conv4layer_output(a,modulation_type=None):
    plt.figure(figsize=(4,3),dpi=600)
    for i in range(100):
        plt.plot(range(124),a[0,0,:,i])
        plt.xticks([])  #去掉横坐标值
        plt.yticks(size=20)
        save_filename='./figure_conv4_output/output%d.png'%i
        plt.savefig(save_filename,dpi=600,bbox_inches='tight')
        plt.tight_layout()
        plt.close()
 

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.get_cmap("Blues"), labels=[],save_filename=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm*100, interpolation='nearest', cmap=cmap)
    #plt.title(title,fontsize=10)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90,size=12)
    plt.yticks(tick_marks, labels,size=12)
    #np.set_printoptions(precision=2, suppress=True)
    for i in range(len(tick_marks)):
        for j in range(len(tick_marks)):
            if i!=j:
                text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10)
            elif i==j:
                if int(np.around(cm[i,j]*100))==100:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=8,color='darkorange')
                else:
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=10,color='darkorange')
            

    plt.tight_layout()
    #plt.ylabel('True label',fontdict={'size':8,})
    #plt.xlabel('Predicted label',fontdict={'size':8,})
    if save_filename is not None:
        plt.savefig(save_filename,dpi=600,bbox_inches = 'tight')
    plt.close()

def calculate_confusion_matrix(Y,Y_hat,classes):
    n_classes = len(classes)
    conf = np.zeros([n_classes,n_classes])
    confnorm = np.zeros([n_classes,n_classes])

    for k in range(0,Y.shape[0]):
        i = list(Y[k,:]).index(1)
        j = int(np.argmax(Y_hat[k,:]))
        conf[i,j] = conf[i,j] + 1

    for i in range(0,n_classes):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    # print(confnorm)

    right = np.sum(np.diag(conf))
    wrong = np.sum(conf) - right
    return confnorm,right,wrong
def calculate_accuracy_each_snr(Y,Y_hat,Z,classes=None):

    Z_array = Z[:,0]
    snrs = sorted(list(set(Z_array)))
    # snrs = np.arange(-20,32,2)
    acc = np.zeros(len(snrs))
    Y_index = np.argmax(Y,axis=1)
    Y_index_hat = np.argmax(Y_hat,axis=1)
    i = 0
    for snr in snrs:
        Y_snr = Y_index[np.where(Z_array == snr)]
        Y_hat_snr = Y_index_hat[np.where(Z_array == snr)]
        acc[i] = np.sum(Y_snr==Y_hat_snr)/Y_snr.shape[0]
        i = i +1
    plt.figure(figsize=(8, 6))
    plt.plot(snrs,acc, label='test_acc')
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on RadioML 2018.01")
    plt.legend()
    plt.grid()
    plt.show()
def calculate_acc_at1snr_from_cm(cm):

    return np.round(np.diag(cm)/np.sum(cm,axis=1),3)

def calculate_acc_cm_each_snr(Y,Y_hat,Z,classes=None,save_figure=True,min_snr = 0):
    Z_array = Z[:,0]
    snrs = sorted(list(set(Z_array)))
    acc = np.zeros(len(snrs))
    acc_mod_snr = np.zeros( (len(classes),len(snrs)) )  #mods*snrs,24*26
    i = 0
    for snr in snrs:
        Y_snr = Y[np.where(Z_array == snr)]
        Y_hat_snr = Y_hat[np.where(Z_array == snr)]
        # plot confusion for each snr
        cm,right,wrong = calculate_confusion_matrix(Y_snr,Y_hat_snr,classes)
        print(min_snr)
        if snr >= min_snr:
            plot_confusion_matrix(cm, title='Confusion matrix at {}db'.format(snr), cmap=plt.cm.Blues, labels=classes,save_filename = 'figure/cm_snr{}.png'.format(snr))
        # cal acc on each snr
        acc[i] = round(1.0*right/(right+wrong),3)
        result = right / (right+wrong)
        with open('acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        print('Accuracy at %ddb:%.2f%s / (%d + %d)' % (snr,100*acc[i],'%',right, wrong))
        acc_mod_snr[:,i] = calculate_acc_at1snr_from_cm(cm)
        i = i +1
    '''
    acc随着snr的变化曲线
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(snrs,acc, label='test_acc')
    # 设置数字标签
    for x, y in zip(snrs,acc):
        plt.text(x, y, y, ha='center', va='bottom', fontsize=8)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy on All Test Data")
    plt.legend()
    plt.grid()
    plt.savefig('figure/acc_overall.png')
    plt.show()
    fd = open('acc_overall_128k_on_512k_wts.dat', 'wb')
    pickle.dump(('128k','512k', acc), fd)
    fd.close()
    '''
    acc随着snr的变化曲线,每个mod一条曲线
    '''
    dis_num = 6
    for g in range(int(np.ceil(acc_mod_snr.shape[0]/dis_num))):
        assert (0 <= dis_num <= acc_mod_snr.shape[0])
        beg_index = g*dis_num
        end_index = np.min([(g+1)*dis_num,acc_mod_snr.shape[0]])
        plt.figure(figsize=(12, 10))
        plt.xlabel("Signal to Noise Ratio")
        plt.ylabel("Classification Accuracy")
        plt.title("Classification Accuracy for Each Mod")
        for i in range(beg_index,end_index):
            plt.plot(snrs, acc_mod_snr[i], label=classes[i])
            # 设置数字标签
            for x, y in zip(snrs, acc_mod_snr[i]):
                plt.text(x, y, y, ha='center', va='bottom', fontsize=8)
        plt.legend()
        plt.grid()
        if save_figure:
            plt.savefig('figure/acc_with_mod_{}.png'.format(g+1))
        plt.show()
    fd = open('acc_for_mod_on_1m_wts.dat', 'wb')
    pickle.dump(('128k','1m', acc_mod_snr), fd)
    fd.close()
    # print(acc_mod_snr)
def main():
    import dataset2016
    import numpy as np
    (mods,snrs,lbl),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),(train_idx,val_idx,test_idx) = \
        dataset2016.load_data()

    one_sample = X_test[0]
    print(np.shape(one_sample))
    print(one_sample[0:2])
    print(np.max(one_sample,axis=1))
    one_sample = np.power(one_sample,2)
    one_sample = np.sqrt(one_sample[0,:]+one_sample[1,:])

    plt.figure()
    plt.title('Training Samples')
    one_sample_t = np.arange(128)
    plt.plot(one_sample_t,one_sample)
    # plt.scatter()
    plt.grid()
    plt.show()

    sum_sample = np.sum(one_sample)
    print(sum_sample)