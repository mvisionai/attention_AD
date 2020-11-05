# -*- coding: utf-8 -*-
"""
Residual Attention Network
"""
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix
import tensorflow as tf
from model.utils import EarlyStopping
from AD_Dataset import Dataset_Import
from model.residual_attention_network import ResidualAttentionNetwork
from hyperparameter import HyperParams as hp
from tqdm import tqdm
import AD_Constants as const
import  clr
import  os
import shutil
import pandas as pd
from sklearn.model_selection import LeaveOneOut,KFold,StratifiedKFold
import matplotlib.pyplot as plt
import  scipy
import  cv2
from  scipy.ndimage.interpolation import  zoom
import nibabel as nib

from skimage.transform import resize


#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def move_file(src_file,act_class):
    try :

        dest_file=None

        if act_class == 0:
            dest_file=const.c_ad_dir



        elif  act_class == 1:
            dest_file=const.c_mci_dir


        elif  act_class ==2:
            dest_file=const.c_nc_dir


        shutil.move(src_file,dest_file)
    except Exception:
        print("Failed  to move file ",src_file)
        raise


def train_validate():

    print("Train Residual Attention Model")
    #train_X, train_y, valid_X, valid_y, test_X, test_y = utils.load_data()

    data_feed=Dataset_Import()

    all_train_data=shuffle(data_feed.all_source_data(augment_data=False))
    #all_train_data_2 = shuffle(data_feed.all_3T_dataset(augment_data=False))
    #all_train_data=shuffle(np.concatenate((all_train_data_1,all_train_data_2) ,axis=0))

    print("len ",len(all_train_data))


    validation_data = shuffle(data_feed.all_main_validate())

    val_length = len(validation_data)



    total_data_length = len(all_train_data)
    num_batches = int(total_data_length / hp.BATCH_SIZE)
    initial_learning_rate=1e-6


    print("building  graph...")
    model = ResidualAttentionNetwork()
    early_stopping = EarlyStopping(limit=150)

    x = tf.placeholder("float", [None,212,260,260, 1])
    t = tf.placeholder("float", [None, 3])

    train_loss = tf.Variable(0.85,trainable=False)
    train_accuracyi = tf.Variable(0.,trainable=False)

    val_loss = tf.Variable(0.87, trainable=False)
    val_accuracy = tf.Variable(0., trainable=False)

    dropout_prob = tf.placeholder("float", None, name='keep_proba')

    is_training = tf.placeholder(tf.bool, shape=())

    steps=tf.Variable(0,trainable=False)
    global_step = tf.Variable(0, trainable=False)



    # learning_rate = tf.train.exponential_decay(initial_learning_rate,
    #                                 steps,
    #                                 num_batches,
    #                                 0.30,
    #                                 staircase=True)

    #learning_rate =initial_learning_rate

    y = model.f_prop(x,is_training=is_training,keep_prop=dropout_prob)


    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t)

    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=t)) #tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-7), reduction_indices=[1])) #
    tf.summary.scalar('training_loss',train_loss)
    tf.summary.scalar('training_accuracy',train_accuracyi)

    tf.summary.scalar('validation_loss', val_loss)
    tf.summary.scalar('validation_accuracy', val_accuracy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)), tf.float32))

    tf_acc,tf_update=tf.metrics.accuracy(labels= tf.argmax(t, 1),predictions=tf.argmax(y, 1))

    train = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(loss)  #1e-3 tf.reduce_mean(loss)
    #train2=SGD(0.0001)
    #valid = tf.argmax(y, 1) clr.cyclic_learning_rate(global_step=global_step, mode='triangular')

    merged_summary = tf.summary.merge_all()
    print("start to train...")

    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    #tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #tf_config=tf.ConfigProto(gpu_options=gpu_options)  #config=tf_config

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_l = tf.local_variables_initializer()
    with tf.Session(config=config) as sess:

        train_summary_writer = tf.summary.FileWriter(hp.train_summary, sess.graph)
        #test_summary_writer = tf.summary.FileWriter(hp.test_summary)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(init_l)

        best_accuracy=0
        best_average_loss= float('inf')

        for epoch in range(hp.NUM_EPOCHS):
            shuffle_data = shuffle(all_train_data)
            counter = 0

            reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            sess.run(reset_metrics_op)

            # train
            train_costs = []
            train_accuracy = []
            summary_eval=None
            for i in tqdm(range(num_batches)):
                # print(i)

                batch_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, shuffle_data[
                                                                                                      counter:counter + hp.BATCH_SIZE])])
                #assign_op = global_step.assign(i)
                #sess.run(assign_op)
                _, _loss,_accuracy,lib_acc,lib_up= sess.run([train, loss,accuracy,tf_acc,tf_update], feed_dict={x: list(batch_images[0:, 0]), t:list(batch_images[0:, 1]), is_training: True,dropout_prob:hp.DROP_OUT})

                train_costs.append(_loss)
                train_accuracy.append(_accuracy)
                counter = (counter + hp.BATCH_SIZE) % total_data_length
                print("Custom_acc ",lib_acc)
                #print("counter ",counter)
            print(" ",end="\n")
            print("Training Loss {:.7f} ".format(np.mean(train_costs)))
            print("Training Acc. {:.7f} ".format(np.mean(train_accuracy)))
            print(" ", end="\n")

            t_loss=tf.assign(train_loss,np.mean(train_costs))
            t_acc=tf.assign(train_accuracyi,np.mean(train_accuracy))


            sess.run([t_loss,t_acc])


            # valid
            valid_costs = []
            val_c = 0
            valid_accuracy = []

            for i in range(int(val_length/hp.BATCH_SIZE)):
                v_batch_images = np.asarray( [data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, validation_data[val_c:val_c + hp.BATCH_SIZE])])
                valid_cost,_vaccuracy = sess.run([loss,accuracy], feed_dict={x:list(v_batch_images[0:, 0]), t:list(v_batch_images[0:, 1]), is_training: False,dropout_prob:1.0})
                #acc_1,acc_total= sess.run(tf.metrics.accuracy(labels=tf.argmax(t,1),predictions=tf.argmax(y,1)))
                #print("checker ",acc_1)
                #valid_predictions.extend(pred)
                valid_accuracy.append(_vaccuracy)
                valid_costs.append(valid_cost)

                val_c = (val_c + hp.BATCH_SIZE) % val_length
                #print("val count ", val_c)

            # f1_score = f1_score(np.argmax(valid_y, 1).astype('int32'), valid_predictions, average='macro')
            #v_accuracy = #accuracy_score(np.argmax(list(v_batch_images[0:, 1]), 1).astype('int32'), valid_predictions)

            #if epoch % 2 == 0:

            print('EPOCH: {epoch}, Training cost: {train_cost}, Validation cost: {valid_cost}, Validation Accuracy: {accuracy} '
                      .format(epoch=epoch+1, train_cost=np.mean(train_costs), valid_cost=np.mean(valid_costs), accuracy=np.mean(valid_accuracy)))

            average_accuracy=np.mean(valid_accuracy)
            average_loss=np.mean(valid_costs)

            v_loss = tf.assign(val_loss,average_loss)
            v_acc = tf.assign(val_accuracy,average_accuracy )
            sess.run([v_loss, v_acc])

            summary_eval=sess.run(merged_summary)

            train_summary_writer.add_summary(summary_eval, global_step=epoch+1)

            if average_accuracy > best_accuracy:

              best_accuracy  =average_accuracy
              best_average_loss=average_loss

              print("saving model at epoch " +str(epoch) +" ......")

              os.mkdir(hp.model_directory+os.sep+"chk_"+str(epoch+1)+"_"+str(round(best_accuracy,2)))
              path_chkp=hp.model_directory+os.sep+"chk_"+str(epoch+1)+"_"+str(round(best_accuracy,2))
              saver = tf.train.Saver()
              saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)

            elif  average_accuracy == best_accuracy and average_loss<best_average_loss:

              print("saving  model at epoch " +str(epoch) +" ......")

              os.mkdir(hp.model_directory+os.sep+"chk_r"+str(epoch+1)+"_"+str(round(best_accuracy,2)))
              path_chkp=hp.model_directory+os.sep+"chk_r"+str(epoch+1)+"_"+str(round(best_accuracy,2))
              saver = tf.train.Saver()
              saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)

            if early_stopping.check(average_loss):

              print("saving final model at epoch " +str(epoch) +" ......")

              os.mkdir(hp.model_directory+os.sep+"chk_final"+str(epoch+1)+"_"+str(round(average_accuracy,2)))
              path_chkp=hp.model_directory+os.sep+"chk_final"+str(epoch+1)+"_"+str(round(average_accuracy,2))
              saver = tf.train.Saver()
              saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)
              break


        print("saving final model at epoch " +str(epoch) +" ......")
        os.mkdir(hp.model_directory+os.sep+"chk_final_S"+str(epoch+1)+"_"+str(round(average_accuracy,2)))
        path_chkp=hp.model_directory+os.sep+"chk_final_S"+str(epoch+1)+"_"+str(round(average_accuracy,2))
        saver = tf.train.Saver()
        saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)









def test_model():






    print("building  graph for testing ...")
    model = ResidualAttentionNetwork()

    x = tf.placeholder("float", [None,212,260,260, 1])
    t = tf.placeholder("float", [None, 3])



    dropout_prob = tf.placeholder("float", None, name='keep_proba')

    is_training = tf.placeholder(tf.bool, shape=())




    y = model.f_prop(x,is_training=is_training,keep_prop=dropout_prob)



    #loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=t))
    predicted_class=tf.argmax(y, dimension=1)
    true_class=tf.argmax(t, dimension=1)
    correct_pred=tf.equal(predicted_class,true_class)


    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf_acc,tf_update=tf.metrics.accuracy(labels= tf.argmax(t, 1),predictions=tf.argmax(y, 1))




    print("start to train...")



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #init_l = tf.local_variables_initializer()

    saver = tf.train.Saver()


    test_c = 0
    valid_accuracy = []

    batch_size=1

    data_feed=Dataset_Import()

    test_data=shuffle(data_feed.all_test_data(augment_data=False))

    test_length = len(test_data)
    saver =tf.train.Saver()
    
    #tf.train.import_meta_graph("{}.meta".format(checkpoint_file))



    with tf.Session() as sess:

        path_res=hp.restore_path+os.sep+"fold_2/chk_final_S150_0.86/fold_2"+os.sep+"model.ckpt-150"

        print("My path ",path_res)

        saver.restore(sess,path_res)
        for i in tqdm(range(int(test_length/batch_size))):

            current_batch=test_data[test_c:test_c + batch_size]
            v_batch_images = np.asarray( [data for data in data_feed.next_batch_combined(batch_size,current_batch )])
            _vaccuracy,y_pred,true_label,y_pred_cls,t_class,decision = sess.run([accuracy,y,t,predicted_class,true_class,correct_pred], feed_dict={x:list(v_batch_images[0:, 0]), t:list(v_batch_images[0:, 1]), is_training: False,dropout_prob:1.0})

            valid_accuracy.append(_vaccuracy)

            test_c = (test_c + batch_size) % test_length

            if  decision[0] :

                print("Fine not complex ")

            else :

               print("Prepearing to move file ",current_batch[0],"class",t_class[0])
               #move_file(current_batch[0][0],t_class[0])

           # print ("Predicted -- Actual---  Pred class  Actual-Class -- Decision",y_pred, " -- ",true_label," ---- ",y_pred_cls," ---- ",t_class," ---- ",decision[0])


        print('EPOCH: {epoch}, Validation Accuracy: {accuracy} '.format(epoch=i+1,accuracy=np.mean(valid_accuracy)))


def select_best_model(src_path=None,limit=100):
     tmp_file_name = os.listdir(src_path)              
     test = [] 
     path=[]                                        
                             
                                                       
     for x in tmp_file_name:  
         
        if x.find("events") < 0:
         
           splits=x.split("_")
         
           if splits[3] !="final":
            
             vals=splits[-1] 
             test.append(float(vals))
             path.append(src_path+os.sep+x)
                                   
     
     if len(test) > limit:
         
         m = min(test)
         
         for i, j in enumerate(test) :
             
             if j == m:
                
                 del_dir=path[i]
                 shutil.rmtree(del_dir)


def reconstruct_latent_code():


    print("Rebuilding AEE  code ...")
    model = ResidualAttentionNetwork()

    mri_input = tf.placeholder("float", [None, 260, 260, 260, 1])
    decoder_input = tf.placeholder("float", [None, 4, 4, 4, 128])

    real_dist = tf.placeholder("float", [None, 4, 4, 4, 128])
    target_input = tf.placeholder("float", [None, 260, 260, 260, 1])


    print("start to train...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # init_l = tf.local_variables_initializer()

    encoder_output = model.encoder(mri_input)
    decoder_output = model.decoder(encoder_output)

    d_real = model.discriminator(real_dist)
    d_fake = model.discriminator(encoder_output, reuse=True)



    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(target_input - decoder_output))

    # Discrimminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))






    data_feed = Dataset_Import()

    all_test_data = data_feed.all_test_data(augment_data=False)


    total_test_length = len(all_test_data)

    BATCH_SIZE=1

    test_batches = int(total_test_length /BATCH_SIZE)


    saver = tf.train.Saver()

    # tf.train.import_meta_graph("{}.meta".format(checkpoint_file))


    val_c = 0

    plot_latent = hp.latent_plots
    plot_recons = hp.recons_plots


    if os.path.isdir(plot_latent) == False:
        os.mkdir(plot_latent)

    if os.path.isdir(plot_recons) == False:
        os.mkdir(plot_recons)


    with tf.Session() as sess:

            path_res = hp.restore_path + os.sep + "aee_trained/chk_150" + os.sep + "model.ckpt-150"

            print("My path ", path_res)

            saver.restore(sess, path_res)


            for v in tqdm(range(test_batches)):
                z_real_dist = np.random.normal(0, 1, (hp.BATCH_SIZE, 4, 4, 4, 128)).astype(np.float32)


                batch_run=all_test_data[ val_c:val_c + BATCH_SIZE]

                mri_group=batch_run[0][1]

                print("mri_path ", mri_group)

                mri_g_name = mri_label(mri_group)



                latent_name=plot_latent+os.sep+mri_g_name+"_l_"+str(v+1)+".jpg"
                real_name = plot_recons + os.sep + mri_g_name + "_o_" + str(v + 1) + ".jpg"
                recons_name=plot_recons+os.sep+mri_g_name+"_r_"+str(v+1)+".jpg"


                v_batch_images = np.asarray([data for data in data_feed.next_batch_combined(BATCH_SIZE,
                                                                                            batch_run,
                                                                                            True)])

                v_target_images = np.asarray([data for data in data_feed.next_batch_combined(BATCH_SIZE,
                                                                                             batch_run,
                                                                                             )])

                av_loss, dv_loss, gv_loss, vrecon_mri, vlatent_code = sess.run(
                    [autoencoder_loss, dc_loss, generator_loss, decoder_output, encoder_output],
                    feed_dict={mri_input: list(v_batch_images[0:, 0]),
                               real_dist: z_real_dist, target_input: list(v_target_images[0:, 0])})


                val_c = (val_c + BATCH_SIZE) % total_test_length


                image_plot=np.asarray(np.squeeze(v_batch_images[0:, 0][0]))


                constructed_plot=np.asarray(np.squeeze(vrecon_mri[0]))

                latent_code=np.asarray(np.squeeze(vlatent_code[0])).flatten().squeeze()
                latent_code=np.resize(latent_code, (20,20,20))



                show_mri_align(np.rot90(image_plot),slice=100,path=real_name)
                show_mri_align(np.rot90(constructed_plot), slice=100, path=recons_name)
                show_mri_align(np.rot90(latent_code), slice=18, path=latent_name)



                print(" ", end="\n")
                print(
                'Validation:{strv}, Reconstruction cost: {recons_cost}, Discriminator cost: {disc_cost}, Generator cost: {gen_cost}'
                    .format(strv="Result", recons_cost=av_loss, disc_cost=dv_loss,
                            gen_cost=gv_loss))



def plot_2():


    print("Rebuilding Classifier code...")




    print("start to train...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # init_l = tf.local_variables_initializer()

    #x = tf.placeholder("float", [None, 212, 260, 260, 1])
    #t = tf.placeholder("float", [None, 2])

    is_training = tf.placeholder(tf.bool, shape=())

    data_feed = Dataset_Import()

    all_test_data = data_feed.all_test_data(augment_data=False)


    total_test_length = len(all_test_data)

    BATCH_SIZE=1

    test_batches = int(total_test_length /BATCH_SIZE)

    saver_meta=hp.restore_path + os.sep + "trained/AD-NC-0.91/AD-NC/fold_1_chk_146_0.9" + os.sep + "model.ckpt-146.meta"

    saver=tf.train.import_meta_graph(saver_meta)




    val_c = 0

    plot_latent = hp.latent_plots
    plot_recons = hp.recons_plots


    # if os.path.isdir(plot_latent) == False:
    #     os.mkdir(plot_latent)
    #
    # if os.path.isdir(plot_recons) == False:
    #     os.mkdir(plot_recons)

    all_variables = tf.trainable_variables()

    print("Variables ", all_variables)

    graph_use = tf.get_default_graph()
    with tf.Session(graph=graph_use) as sess:

            path_res = hp.restore_path + os.sep + "trained/AD-NC-0.91/AD-NC/fold_1_chk_146_0.9" + os.sep + "model.ckpt-146"

            print("My path ", tf.trainable_variables())

            saver.restore(sess, path_res)



            for v in tqdm(range(test_batches)):




                batch_run=all_test_data[ val_c:val_c + BATCH_SIZE]


                mri_group=batch_run[0][1]


                v1_run=graph_use.get_tensor_by_name('residual_block_3/conv1/kernel:0')
                print("mri_path ", mri_group)

                mri_g_name = mri_label(mri_group)



                v_target_images = np.asarray([data for data in data_feed.next_batch_combined(BATCH_SIZE,
                                                                                             batch_run,
                                                                                             )])

                res=sess.run(v1_run.out)



                #
                # valid_cost, _vaccuracy = sess.run(feed_dict={"x:0": list(v_batch_images[0:, 0]),
                #                                                                "t:0": list(v_batch_images[0:, 1]),
                #                                                                "is_training:0": False, "dropout_prob:0": 1.0})


                val_c = (val_c + BATCH_SIZE) % total_test_length











def mri_label(mri_group):

    mri_group = int (mri_group)

    if mri_group == 0:

        mri_g_name = "AD"
    elif mri_group == 1:

        mri_g_name = "NC"
    elif mri_group == 2:

        mri_g_name = "MCI"


    return  mri_g_name

def show_mri_align(mri,slice,path=None):

  plt.imshow(mri[:, :,slice], cmap='gray')
  plt.axis("off")
  #plt.imshow(mri[:, :,110], cmap='gray')
  plt.show()

  #plt.imsave(path,mri[:, :,slice],cmap="gray")


def train_cross_validate():

    print("Train Residual Attention Model")
    #train_X, train_y, valid_X, valid_y, test_X, test_y = utils.load_data()
    

    data_feed=Dataset_Import()

    all_train_data=shuffle(data_feed.all_source_data(augment_data=False))

    #all_train_data_2 = shuffle(data_feed.all_3T_dataset(augment_data=False))
    #all_train_data=shuffle(np.concatenate((all_train_data_1,all_train_data_2) ,axis=0))


    
    X_data=all_train_data[:,0:1]
    Y_data=all_train_data[:,1]


    print("len ",len(all_train_data))


   

    total_data_length = len(all_train_data)
   
    initial_learning_rate=1e-6


    print("building  graph...")
    model = ResidualAttentionNetwork()


    mri_inputs = tf.placeholder("float", [None,260,260,260, 1],name="mri_input")
    label = tf.placeholder("float", [None, 2],name="mri_label")

    train_loss = tf.Variable(0.85,trainable=False)
    train_accuracyi = tf.Variable(0.,trainable=False)

    #val_loss = tf.Variable(0.87, trainable=False)
    #val_accuracy = tf.Variable(0., trainable=False)

    dropout = tf.placeholder("float", None, name='keep_proba')

    training_status = tf.placeholder(tf.bool, shape=())



    y = model.f_prop(mri_inputs,is_training=training_status,keep_prop=dropout)




    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=label)) #tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-7), reduction_indices=[1])) #
    tf.summary.scalar('loss',train_loss)
    tf.summary.scalar('accuracy',train_accuracyi)
#
    #tf.summary.scalar('loss', val_loss)
    #tf.summary.scalar('accuracy', val_accuracy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(label, 1)), tf.float32))

    #tf_acc,tf_update=tf.metrics.accuracy(labels= tf.argmax(label, 1),predictions=tf.argmax(y, 1))
    
    #tf_rms,tf_update_r=tf.metrics.root_mean_squared_error(labels= tf.argmax(label, 1),predictions=tf.argmax(y, 1))
    

    train = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(loss)


    merged_summary = tf.summary.merge_all()
    print("start to train...")


    #graph_main=tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_l = tf.local_variables_initializer()
    with tf.Session(config=config) as sess:

        print("Graph variables ", tf.all_variables(), end="\n")


        k_fold = StratifiedKFold(n_splits=3,shuffle=False)
        k_counter=1
        
       
        summary_path=hp.model_directory+os.sep+"summary"
        summary_train=summary_path+os.sep+"train"
        summary_val=summary_path+os.sep+"validate"
        
        folds_summary=hp.model_directory+os.sep+"k_folds"
        
        
        class_dir='-'.join(const.classify_group)
        
        
        class_dirs=hp.model_directory+os.sep+class_dir
        
        if os.path.isdir(class_dirs) == False :
            
            os.mkdir(class_dirs)
            
            
        if os.path.isdir(summary_train) == False :
            
            os.mkdir(summary_train)   
            
        if os.path.isdir(summary_val) == False :
            
            os.mkdir(summary_val) 
            
        #train_summary_writer = tf.summary.FileWriter(hp.model_directory+os.sep+class_dir, sess.graph)    
        
     
            
        summary_writer2= tf.summary.FileWriter(summary_train, sess.graph) 
        summary_writer3= tf.summary.FileWriter(summary_val, sess.graph) 
        
        
       
        
        for train_indices, test_indices in k_fold.split(X_data,Y_data):
            
                init = tf.global_variables_initializer()
                sess.run(init)
                sess.run(init_l)
                early_stopping = EarlyStopping(limit=450)
              
                val_remove=[]
                train_remove = []
                
                best_accuracy=0
                best_average_loss= float('inf')
            
                print("Cross Validation Fold On ",k_counter )
                
                par_dir="fold_"+str(k_counter)
                
                f_path=folds_summary+os.sep+par_dir
                
                os.mkdir(f_path)
                
                f_path_t = f_path+os.sep+"train"
                f_path_v = f_path+os.sep+"validate"
                
                os.mkdir(f_path_t)
                os.mkdir(f_path_v)
                
                
                summary_writer_ft= tf.summary.FileWriter(f_path_t, sess.graph) 
                summary_writer_fv= tf.summary.FileWriter(f_path_v, sess.graph) 

                                 
                chk_path=hp.model_directory+os.sep+class_dir+os.sep+par_dir+"_"
                
                train_data = all_train_data[train_indices]
                validation_data =all_train_data[test_indices]


                #avoid augmentation in validation data

                # v_count=0
                # v_list=[]
                # for datas in validation_data:
                #
                #
                #      if datas[0].find("_aug.nii") > -1:
                #          val_remove.append(list(datas))
                #          v_list.append(v_count)
                #      v_count=v_count+1
                #
                # data_checker=len(val_remove)
                #
                # validation_data= np.delete(validation_data,v_list,0)
                #
                #
                #
                # t_count=0
                # t_list=[]
                #
                # for datat in train_data :
                #
                #     if datat[0].find("/AD/") > -1 and datat[0].find("_aug.nii") <0:
                #
                #         if  data_checker  ==  len(train_remove) :
                #             break
                #
                #         train_remove.append(list(datat))
                #         t_list.append(t_count)
                #     t_count=t_count+1
                #
                # train_data=np.delete(train_data,t_list,0)
                #
                #
                #
                # validation_data=np.append(validation_data,train_remove,axis=0)
                # train_data = np.append(train_data,val_remove, axis=0)
                #
                # val_remove=[]
                # train_remove=[]
                # t_list=[]
                # v_list=[]



                val_length = len(validation_data)
                
                num_batches = int(len(train_data) / hp.BATCH_SIZE)
                
                fold_train_acc=[]
                fold_val_acc=[]
                fold_train_loss=[]
                fold_val_loss=[]

                # control images with aligned code

                graph2 = tf.Graph()
                res_sess = tf.Session(graph=graph2)

                with graph2.as_default():


                    models = ResidualAttentionNetwork()

                    mri_input = tf.placeholder("float", [None, 260, 260, 260, 1])
                    decoder_input = tf.placeholder("float", [None, 4, 4, 4, 128])

                    real_dist = tf.placeholder("float", [None, 4, 4, 4, 128])
                    target_input = tf.placeholder("float", [None, 260, 260, 260, 1])

                    encoder_output = models.encoder(mri_input)
                    decoder_output = models.decoder(encoder_output)

                    # d_real = models.discriminator(real_dist)
                    # d_fake = models.discriminator(encoder_output, reuse=True)
                    #
                    # # Autoencoder loss
                    # autoencoder_loss = tf.reduce_mean(tf.square(target_input - decoder_output))
                    #
                    # # Discrimminator Loss
                    # dc_loss_real = tf.reduce_mean(
                    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
                    # dc_loss_fake = tf.reduce_mean(
                    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
                    # dc_loss = dc_loss_fake + dc_loss_real
                    #
                    # # Generator loss
                    # generator_loss = tf.reduce_mean(
                    #     tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

                    #saver_meta =  hp.restore_path + os.sep + "aee_trained/chk_150" + os.sep + "model.ckpt-150.meta"
                    #saver_recons = tf.train.import_meta_graph(saver_meta,clear_devices=True)
                    z_real_dist = np.random.normal(0, 1, (hp.BATCH_SIZE, 4, 4, 4, 128)).astype(np.float32)
                    saver_recons = tf.train.Saver()
                    path_recons =  hp.restore_path + os.sep + "aee_trained/chk_150" + os.sep + "model.ckpt-150"
                    saver_recons.restore(res_sess, path_recons)



                for epoch in range(hp.NUM_EPOCHS):

                    shuffle_data = shuffle(train_data)
                  
                    counter = 0

                    reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
                    sess.run(reset_metrics_op)

                    # train
                    train_costs = []
                    train_accuracy = []
                   
                    for i in tqdm(range(num_batches)):
                        # print(i)

                        batch_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, shuffle_data[
                                                                                                              counter:counter + hp.BATCH_SIZE])])

                        vrecon_mri= res_sess.run(
                            decoder_output,
                            feed_dict={mri_input: list(batch_images[0:, 0]),
                                       real_dist: z_real_dist, target_input: list(batch_images[0:, 0])})

                        #,lib_acc,lib_up=,tf_acc,tf_update]

                        _, _loss,_accuracy= sess.run([train, loss,accuracy], feed_dict={mri_inputs: vrecon_mri, label:list(batch_images[0:, 1]), training_status: True,dropout:hp.DROP_OUT})

                        train_costs.append(_loss)
                        train_accuracy.append(_accuracy)
                        counter = (counter + hp.BATCH_SIZE) % total_data_length
                        #print("Custom_acc ",lib_acc)
                        #print("counter ",counter)
                    print(" ",end="\n")
                    
                    average_tran_acc=np.mean(train_accuracy)
                    average_train_loss=np.mean(train_costs)
                    print("Training Loss {:.7f} ".format(average_train_loss))
                    print("Training Acc. {:.7f} ".format(average_tran_acc))
                    print(" ", end="\n")

                    t_loss=tf.assign(train_loss,np.mean(train_costs))
                    t_acc=tf.assign(train_accuracyi,np.mean(train_accuracy))


                    sess.run([t_loss,t_acc])


                    step_epochs=epoch+1+(k_counter-1)*hp.NUM_EPOCHS    
                    
                    
                    summary_eval=sess.run(merged_summary)
                    summary_writer2.add_summary(summary_eval, global_step=step_epochs)
                    summary_writer2.flush()
                    
                    summary_writer_ft.add_summary(summary_eval, global_step=epoch+1)
                    summary_writer_ft.flush()


                    # valid
                    valid_costs = []
                    val_c = 0
                    valid_accuracy = []
                    

                    for i in range(int(val_length/hp.BATCH_SIZE)):
                        v_batch_images = np.asarray( [data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, validation_data[val_c:val_c + hp.BATCH_SIZE])])

                        varecon_mri = res_sess.run(
                            decoder_output,
                            feed_dict={mri_input: list(v_batch_images[0:, 0]),
                                       real_dist: z_real_dist, target_input: list(v_batch_images[0:, 0])})


                        valid_cost,_vaccuracy = sess.run([loss,accuracy], feed_dict={mri_inputs:varecon_mri, label:list(v_batch_images[0:, 1]), training_status: False,dropout:1.0})


                        valid_accuracy.append(_vaccuracy)
                        valid_costs.append(valid_cost)
                       

                        val_c = (val_c + hp.BATCH_SIZE) % val_length


                    print('EPOCH: {epoch}, Training cost: {train_cost}, Validation cost: {valid_cost}, Validation Accuracy: {accuracy}'
                              .format(epoch=epoch+1, train_cost=np.mean(train_costs), valid_cost=np.mean(valid_costs), accuracy=np.mean(valid_accuracy)))

                    average_accuracy=np.mean(valid_accuracy)
                    average_loss=np.mean(valid_costs)
                    
    


                    v_loss = tf.assign(train_loss,average_loss)
                    v_acc = tf.assign(train_accuracyi,average_accuracy )
                    sess.run([v_loss, v_acc])
                    
                    
                    summary_eval=sess.run(merged_summary)
                    summary_writer3.add_summary(summary_eval, global_step=step_epochs)           
                    summary_writer3.flush()
                    
                    
                    summary_writer_fv.add_summary(summary_eval, global_step=epoch+1)
                    summary_writer_fv.flush()


                   
                    
                    print("Epoch ",step_epochs)
                    
                    
                    
                    #train_summary_writer.add_summary(summary_eval, global_step=step_epochs)
                    
                    
                    # Create a new Summary object with your measure
                    
                   
                   
                    #summary = tf.Summary()
                 
                   
                   # summary.value.add(tag="Training_Accuracy", simple_value=average_tran_acc)
                    #summary.value.add(tag="Training_Loss", simple_value=average_train_loss)
                    #summary.value.add(tag="Validation_Accuracy", simple_value=average_accuracy)
                    #summary.value.add(tag="Validation_Loss", simple_value=average_loss)
                    
                   
                    
                   
                   
                    
                    # Add it to the Tensorboard summary writer
                    # Make sure to specify a step parameter to get nice graphs over time
                    
                    
                    
                    
                    
                    print(" ",end="\n")
                    
                    print("saving model at epoch " +str(epoch+1) +" ......")

                    os.mkdir(chk_path+"chk_"+str(epoch+1)+"_"+str(round(average_accuracy,2)))
                    path_chkp=chk_path+"chk_"+str(epoch+1)+"_"+str(round(average_accuracy,2))
                    saver = tf.train.Saver()
                    saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)
                    
                    
                    
                    
                    if average_accuracy > best_accuracy:
                        
                      print("Val Accuracy Improved from ",best_accuracy, " to ",average_accuracy," at epoch "+str(epoch+1))

                      best_accuracy  =average_accuracy
                      best_average_loss=average_loss
                      
                        
#
#                    elif  average_accuracy == best_accuracy and average_loss<best_average_loss:
#
#                      print("saving  model at epoch " +str(epoch+1) +" ......")
#
#                      os.mkdir(chk_path+"chk_r"+str(epoch+1)+"_"+str(round(best_accuracy,2)))
#                      path_chkp=chk_path+"chk_r"+str(epoch+1)+"_"+str(round(best_accuracy,2))
#                      saver = tf.train.Saver()
#                      saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)
                    
                    select_best_model(class_dirs)
                    if early_stopping.check(average_loss):

                      print("saving final model at epoch " +str(epoch+1) +" ......")

                      os.mkdir(chk_path+"chk_final"+str(epoch+1)+"_"+str(round(average_accuracy,2)))
                      path_chkp=chk_path+"chk_final"+str(epoch+1)+"_"+str(round(average_accuracy,2))
                      saver = tf.train.Saver()
                      saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)
                      break


                print("saving final model at epoch " +str(epoch+1) +" ......")
                os.mkdir(chk_path+"chk_final_S"+str(epoch+1)+"_"+str(round(average_accuracy,2)))
                path_chkp=chk_path+"chk_final_S"+str(epoch+1)+"_"+str(round(average_accuracy,2))
                saver = tf.train.Saver()
                saver.save(sess, path_chkp+os.sep+"model.ckpt", global_step=epoch+1)
                fold_val_acc.append(best_accuracy)
                fold_val_loss.append(best_average_loss)
                fold_train_acc.append(average_tran_acc)
                fold_train_loss.append(average_train_loss)

                 
                print(" ",end="\n")
                print('Fold: {fold},Training Acc:{train_ac},Training cost: {train_cost}, Validation cost: {valid_cost}, Validation Accuracy: {accuracy}'
                              .format(fold=k_counter,train_ac=average_tran_acc,train_cost=average_train_loss, valid_cost=best_average_loss, accuracy=best_accuracy))             
                
                pd_f=pd.DataFrame({"train_acc":[average_tran_acc],"train_loss":[average_train_loss],"val_acc":[best_accuracy],"val_loss":[best_average_loss]})
                pd_f.to_csv(hp.fold_path+os.sep+'output_csv.csv', mode='a', header=False,encoding='utf8')
                k_counter=k_counter+1





def AAE():


    print("Train Adverserial Autoencoder Model")


    data_feed = Dataset_Import()

    all_train_data = shuffle(data_feed.all_source_data(augment_data=False))



    all_test_data = shuffle(data_feed.all_test_data(augment_data=False))




    #X_data = all_train_data[:, 0:1]
    #Y_data = all_train_data[:, 1]

    total_data_length = len(all_train_data)

    total_test_length = len(all_test_data)

    print("Number of Training Data : ", total_data_length)

    num_batches = int(total_data_length / hp.BATCH_SIZE)\


    test_batches = int(total_test_length / hp.BATCH_SIZE)

    initial_learning_rate = 1e-6

    print("Building  graph...")

    model = ResidualAttentionNetwork()

    mri_input = tf.placeholder("float", [None, 260, 260, 260, 1])
    decoder_input = tf.placeholder("float", [None, 4, 4, 4, 128])
    real_dist = tf.placeholder("float", [None, 4, 4, 4, 128])
    target_input = tf.placeholder("float", [None, 260, 260, 260, 1])

    auto_loss_t = tf.Variable(0., trainable=False)

    disc_loss_t = tf.Variable(0., trainable=False)


    tf.summary.scalar('Autoencoder_loss', auto_loss_t)
    tf.summary.scalar('Discriminator_loss', disc_loss_t)


    encoder_output = model.encoder(mri_input)
    decoder_output = model.decoder(encoder_output)

    d_real = model.discriminator(real_dist)
    d_fake = model.discriminator(encoder_output, reuse=True)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(target_input - decoder_output))

    # Discrimminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()

    print("Variables ",all_variables)
    dc_var = [var for var in all_variables if 'Discriminator' in var.name]
    en_var = [var for var in all_variables if 'Encoder' in var.name]
    dec_var = [var for var in all_variables if 'Decoder' in var.name]

    auto_enc_var = dec_var + en_var


    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,
                                                   beta1=0.9).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,
                                                     beta1=0.9).minimize(dc_loss, var_list=dc_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,
                                                 beta1=0.9).minimize(generator_loss, var_list=en_var)

    merged_summary = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init_l = tf.local_variables_initializer()
    with tf.Session(config=config) as sess:


        init = tf.global_variables_initializer()
        sess.run(init)

        summary_path = hp.autoenc_directory
        summary_train = summary_path + os.sep + "train"
        summary_val = summary_path + os.sep + "validate"

        summary_writer2 = tf.summary.FileWriter(summary_train, sess.graph)
        summary_writer3 = tf.summary.FileWriter(summary_val, sess.graph)

        chk_path = hp.model_directory + os.sep + "aee_trained"

        for epoch in range(hp.NUM_EPOCHS):

            shuffle_data = shuffle(all_train_data)

            counter = 0

            train_re_costs = []
            train_dis_cost = []
            train_gen_cost = []

            for i in tqdm(range(num_batches)):
                # print(i)

                z_real_dist = np.random.normal(0,1,(hp.BATCH_SIZE,4,4,4,128)).astype(np.float32)

                batch_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, shuffle_data[
                                                                                                         counter:counter + hp.BATCH_SIZE],noise=True)])

                target_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE, shuffle_data[
                                                                                                         counter:counter + hp.BATCH_SIZE])])



                sess.run(autoencoder_optimizer, feed_dict={mri_input:list(batch_images[0:, 0]),target_input:list(target_images[0:, 0])})

                sess.run(discriminator_optimizer,feed_dict={mri_input:list(batch_images[0:, 0]),real_dist: z_real_dist})

                sess.run(generator_optimizer, feed_dict={mri_input:list(batch_images[0:, 0])})

                a_loss, d_loss, g_loss,recon_mri,latent_code= sess.run([autoencoder_loss, dc_loss, generator_loss,decoder_output,encoder_output],
                                                           feed_dict={mri_input: list(batch_images[0:, 0]),
                                                                      real_dist: z_real_dist,target_input:list(target_images[0:, 0])})

                train_re_costs.append(a_loss)
                train_dis_cost.append(d_loss)
                train_gen_cost.append(g_loss)
                counter = (counter + hp.BATCH_SIZE) % total_data_length

            batch_images=None
            target_images=None


            print(" ", end="\n")
            print( 'EPOCH: {epoch}, Reconstruction cost: {recons_cost}, Discriminator cost: {disc_cost}, Generator cost: {gen_cost}'
                .format(epoch=epoch + 1, recons_cost=np.mean(train_re_costs), disc_cost=np.mean(train_dis_cost),gen_cost=np.mean(train_gen_cost)))

            sa_loss = tf.assign(auto_loss_t, np.mean(train_re_costs))
            sd_loss = tf.assign(disc_loss_t, np.mean(train_dis_cost))

            sess.run([sa_loss, sd_loss])

            summary_eval = sess.run(merged_summary)
            summary_writer2.add_summary(summary_eval, global_step=epoch+1)
            summary_writer2.flush()

            test_re_costs = []
            test_dis_cost = []
            test_gen_cost = []
            val_c = 0

            for v in tqdm(range(test_batches)):

                v_batch_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE,
                                                                                            all_test_data[
                                                                                            val_c:val_c + hp.BATCH_SIZE],True)])

                v_target_images = np.asarray([data for data in data_feed.next_batch_combined(hp.BATCH_SIZE,
                                                                                            all_test_data[
                                                                                            val_c:val_c + hp.BATCH_SIZE],
                                                                                            )])

                av_loss, dv_loss, gv_loss,vrecon_mri,vlatent_code = sess.run([autoencoder_loss, dc_loss, generator_loss,decoder_output,encoder_output],
                                                  feed_dict={mri_input: list(v_batch_images[0:, 0]),
                                                             real_dist: z_real_dist,target_input:list(v_target_images[0:, 0])})

                test_re_costs.append(av_loss)
                test_dis_cost.append(dv_loss)
                test_gen_cost.append(gv_loss)
                val_c = (val_c + hp.BATCH_SIZE) % total_test_length

            print(" ", end="\n")
            print(
                'Validation:{strv}, Reconstruction cost: {recons_cost}, Discriminator cost: {disc_cost}, Generator cost: {gen_cost}'
                .format(strv= "Result", recons_cost=np.mean(test_re_costs), disc_cost=np.mean(test_dis_cost),
                        gen_cost=np.mean(test_gen_cost)))


            sav_loss = tf.assign(auto_loss_t,np.mean(test_re_costs))
            sdv_loss = tf.assign(disc_loss_t, np.mean(test_dis_cost))
            sess.run([sav_loss ,sdv_loss])

            summary_eval = sess.run(merged_summary)
            summary_writer3.add_summary(summary_eval, global_step=epoch+1)
            summary_writer3.flush()
            v_batch_images=None
            v_target_images=None


            os.mkdir(chk_path +os.sep+ "chk_" + str(epoch + 1) )
            path_chkp = chk_path +os.sep+ "chk_" + str(epoch + 1)
            saver = tf.train.Saver()





        #print(saver.saver_def.filename_tensor_name,"---",saver.saver_def.restore_op_name)
            saver.save(sess, path_chkp + os.sep + "model.ckpt", global_step=epoch + 1)



def plot_activation():
    print("Train Residual Attention Model")
    # train_X, train_y, valid_X, valid_y, test_X, test_y = utils.load_data()

    data_feed = Dataset_Import()



    initial_learning_rate = 1e-6

    print("building  graph...")
    model = ResidualAttentionNetwork()

    x = tf.placeholder("float", [None, 260, 260, 260, 1])
    t = tf.placeholder("float", [None, 2])



    is_training = tf.placeholder(tf.bool, shape=())

    dropout_prob = tf.placeholder("float", None, name='keep_proba')

    y = model.f_prop(x, is_training=is_training, keep_prop=dropout_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,
                                                                     labels=t))  # tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-7), reduction_indices=[1])) #

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)), tf.float32))



    train = tf.train.AdamOptimizer(learning_rate=initial_learning_rate).minimize(loss)

    #/HDD/public/data/ADNI_CROSS/train/AD/130_S_4971.nii 116_S_4732.nii  023_S_0083.nii
    #//ARVID CHECK '/HDD/public/data/ADNI_CROSS/train/NC/019_S_5012.nii','0','1'
    all_test_data =[['sample_data/002_S_0413.nii','1','1']]
    #[['/HDD/public/data/ADNI_CROSS/train/AD/023_S_0078.nii','1','1']] #data_feed.all_test_data(augment_data=False)

    total_test_length = len(all_test_data)



    BATCH_SIZE = 1

    test_batches = int(total_test_length / BATCH_SIZE)



    saver = tf.train.Saver()


    print("start to train...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True



    graph=tf.get_default_graph()
    with tf.Session(config=config) as sess:



                val_c=0

                path_res =hp.restore_path + os.sep + "AD-NC/fold_3_chk_113_0.82" + os.sep + "model.ckpt-113"
                #path_res="/home/ben/PycharmProjects/train_session/AD-MCI-AL-1/AD-MCI/fold_2_chk_144_0.87/model.ckpt-144"
                #path_res =hp.restore_path + os.sep + "AD-MCI/fold_1_chk_135_0.86" + os.sep + "model.ckpt-135"

                print("My path ", path_res)

                saver.restore(sess, path_res)


                print("Graph variables ",tf.all_variables(),end="\n")


                print("My collections ",tf.get_default_graph().get_all_collection_keys())


                actiava_dir=hp.activation_dir+os.sep+"AD-MCI-ACT"

                graph2 = tf.Graph()
                res_sess = tf.Session(graph=graph2)

                with graph2.as_default():
                    models = ResidualAttentionNetwork()

                    mri_input = tf.placeholder("float", [None, 260, 260, 260, 1])
                    decoder_input = tf.placeholder("float", [None, 4, 4, 4, 128])

                    real_dist = tf.placeholder("float", [None, 4, 4, 4, 128])
                    target_input = tf.placeholder("float", [None, 260, 260, 260, 1])

                    encoder_output = models.encoder(mri_input)
                    decoder_output = models.decoder(encoder_output)


                    z_real_dist = np.random.normal(0, 1, (hp.BATCH_SIZE, 4, 4, 4, 128)).astype(np.float32)
                    saver_recons = tf.train.Saver()
                    path_recons = hp.restore_path + os.sep + "aee_trained/chk_150" + os.sep + "model.ckpt-150"
                    saver_recons.restore(res_sess, path_recons)



                for v in tqdm(range(int(test_batches))):

                    batch_run=all_test_data[val_c:val_c + BATCH_SIZE]

                    print(val_c)

                    mri_group = batch_run[0][1]

                    print(batch_run)

                    if mri_group == '0':
                         val_c = (val_c + BATCH_SIZE) % total_test_length
                         print("skipp")
                         continue

                    mri_path=batch_run[0][0]

                    print("mri_path ",mri_group,"-" ,mri_path)

                    mri_g_name = mri_label(mri_group)


                    recons_name = actiava_dir + os.sep + mri_g_name + "_act_" + str(v + 1) + ".jpg"

                    v_batch_images = np.asarray([data for data in data_feed.next_batch_combined(BATCH_SIZE,batch_run
                                                                                                )])


                    #get reconstructed error
                    vrecon_mri = res_sess.run(
                        decoder_output,
                        feed_dict={mri_input: list(v_batch_images[0:, 0]),
                                   real_dist: z_real_dist, target_input: list(v_batch_images[0:, 0])})


                    #print(graph.get_collection("conv_ouput3"))

                    pict,last_conv,before_conv,prediction_weights,prediction=sess.run(
                                            [graph.get_collection("conv_ouput2"),
                                             graph.get_tensor_by_name("residual_module_4/add:0"),
                                             graph.get_collection("avg_collection"),
                                             graph.get_tensor_by_name("dense_1/kernel:0"),y],
                                                                          feed_dict={x:vrecon_mri,
                                                                                   t: list(v_batch_images[0:, 1]),
                                                                                   is_training: False,dropout_prob:1.0})







                    predicted_index=np.argmax(prediction)

                    activated_areas = np.squeeze(np.asarray(last_conv))

                    attended_weights=np.asarray(prediction_weights)

                    attended_weights_target = attended_weights[:,predicted_index]

                    print("alst conv shape",activated_areas.shape)


                    #new_dimen=resize(activated_areas,(240, 240, 240,128))

                    cam_output=np.dot(activated_areas,attended_weights_target)

                    print("lst conv shape", cam_output.shape)

                    affine=np.eye(4)

                    save_original_cam=nib.Nifti1Image(cam_output,affine)

                    nib.save(save_original_cam,"ad_vrs_mci_activation.nii.gz")

                    resize_cam=resize(np.asarray(cam_output), (260, 260, 260))

                    save_roriginal_cam = nib.Nifti1Image(resize_cam, affine)

                    nib.save(save_roriginal_cam, "resize_ad_vrs_mci_activation.nii.gz")

                    cams = np.rot90(resize_cam)

                    #cams = np.rot90(cam_output)

                    reconstruct_output=np.squeeze(np.asarray(vrecon_mri[0]))

                    reconsc=np.rot90(reconstruct_output)

                    save_reconstruct_au = nib.Nifti1Image(reconstruct_output, affine)

                    nib.save(save_reconstruct_au,"new_ad_space_after_autoencoder.nii.gz")

                    original = np.rot90(np.squeeze(np.asarray(v_batch_images[0:, 0][0])))

                    print("shape ",np.squeeze(np.asarray(vrecon_mri[0])).shape)


                    plot_attended_regions( reconsc,cams,135,"")





                    # for index,weight in enumerate(attended_weights_target):
                    #
                    #      cam+=weight*activated_areas[:,:,:,index]


                    val_c = (val_c + BATCH_SIZE) % total_test_length




def plot_CAM(mri,CAM,slice):


    fig, ax = plt.subplots()
    # plot image
    ax.imshow(mri[:, :,slice], alpha=0.9)
    # get class activation map

    # plot class activation map
    ax.imshow(CAM[:, :,slice], cmap='jet', alpha=0.8)

    # obtain the predicted ImageNet category
    ax.set_title("sample")
    plt.show()



def plot_attended_regions(mri,cam,slice,path=None):


  template=np.rot90(Dataset_Import.load_dataset("MNI152_T1_1mm_Brain.nii.gz"))

  print("Template shape ",template.shape)



  fig , ax= plt.subplots()

  #ax.imshow(mri[:, :, 135], cmap="gray", alpha=0.9)
  ax.imshow(cam[:, :, slice], alpha=0.4, cmap="jet")

  #ax.imshow(mri[:, :, slice], alpha=0.9, cmap="gray")




  ax.set_title("Class Activation MAp "+str(path))
  #ax.axis("off")

  plt.show()

  #plt.imsave(path, mri[:, :, slice], cmap="inferno")

  #inferno  viridis



def activation_recode(layer_names,activations,images_per_row ):


    for layer_name, layer_activation in zip(layer_names, activations):
        # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


if __name__ == "__main__":

    #plot_2()
    #plot_activation()
    #reconstruct_latent_code()
    AAE()
    #train_cross_validate()
   #train_validate()
    #test_model()
  # d= select_best_model("/home/arvid/Documents/Ben/codes/PycharmProjects/attention_B/attention_residual/trained_models/AD-MCI-NC")
   #print(d)











