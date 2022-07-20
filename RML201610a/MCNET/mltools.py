import matplotlib
#matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 
import numpy as np
import pickle

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
    plt.figure(figsize=(4, 3),dpi=600)
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
                    text=plt.text(j,i,int(np.around(cm[i,j]*100)),ha="center",va="center",fontsize=7,color='darkorange')
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