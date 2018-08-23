import tensorflow as tf
import numpy as np
import Weibo_model
import os

def main(_):
    print 'read word embedding......'
    embedding=np.load('./data/weibo_vector.npy')
    print 'read ner train data......'
    train_word=np.load('./data/weibo_train_word.npy')
    train_label=np.load('./data/weibo_train_label.npy')
    train_length=np.load('./data/weibo_train_length.npy')
    print 'read cws train data......'
    train_cws_word=np.load('./data/weibo_cws_word.npy')
    train_cws_label=np.load('./data/weibo_cws_label.npy')
    train_cws_length=np.load('./data/weibo_cws_length.npy')
    setting = Weibo_model.Setting()
    task_ner=[]
    task_cws=[]
    for i in range(setting.batch_size):
        task_ner.append([1,0])
        task_cws.append([0,1])

    with tf.Graph().as_default():
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess=tf.Session(config=config)
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('ner_model',reuse=None,initializer=initializer):
                m=Weibo_model.TransferModel(setting,tf.cast(embedding,tf.float32),adv=True,is_train=True)
                m.multi_task()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            global_step1 = tf.Variable(0, name="global_step1", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            if setting.clip>0:
                grads, vs=zip(*optimizer.compute_gradients(m.loss))
                grads,_ =tf.clip_by_global_norm(grads,clip_norm=setting.clip)
                train_op=optimizer.apply_gradients(zip(grads,vs),global_step)
                train_op1 = optimizer.apply_gradients(zip(grads, vs), global_step1)
            else:
                train_op=optimizer.minimize(m.loss, global_step)
                train_op1 = optimizer.minimize(m.loss, global_step1)

            sess.run(tf.initialize_all_variables())
            saver=tf.train.Saver(max_to_keep=None)
            save_path = './ckpt/lstm+crf'
            for one_epoch in range(setting.num_epoches):
                temp_order=range(len(train_word))
                temp_order_cws=range(len(train_cws_word))
                np.random.shuffle(temp_order)
                np.random.shuffle(temp_order_cws)
                for i in range(len(temp_order)//setting.batch_size):
                    for j in range(2):
                        if j==0:
                            temp_word = []
                            temp_label = []
                            temp_label_=[]
                            temp_length = []
                            temp_input_index = temp_order[i * setting.batch_size:(i + 1) * setting.batch_size]
                            temp_input_index1 = temp_order_cws[i * setting.batch_size:(i + 1) * setting.batch_size]
                            for k in range(len(temp_input_index)):
                                temp_word.append(train_word[temp_input_index[k]])
                                temp_label.append(train_label[temp_input_index[k]])
                                temp_label_.append(train_cws_label[temp_input_index1[k]])
                                temp_length.append(train_length[temp_input_index[k]])
                            feed_dict={}
                            feed_dict[m.input]=np.asarray(temp_word)
                            feed_dict[m.label]=np.asarray(temp_label)
                            feed_dict[m.label_]=np.asarray(temp_label_)
                            feed_dict[m.sent_len]=np.asarray(temp_length)
                            feed_dict[m.is_ner]=1
                            feed_dict[m.task_label]=np.asarray(task_ner)
                            _, step, loss= sess.run([train_op,global_step,m.ner_loss],feed_dict)
                            if step % 20 ==0:
                                temp = "step {},loss {}".format(step, loss)
                                print temp
                            current_step = step
                            if current_step % 120 == 0 and current_step > 2000 and current_step < 10000:
                                saver.save(sess, save_path=save_path, global_step=current_step)
                        else:
                            temp_cws_word = []
                            temp_cws_label = []
                            temp_cws_label_=[]
                            temp_cws_length = []
                            temp_input_index = temp_order_cws[i * setting.batch_size:(i + 1) * setting.batch_size]
                            temp_input_index1 = temp_order[i * setting.batch_size:(i + 1) * setting.batch_size]
                            for k in range(len(temp_input_index)):
                                temp_cws_word.append(train_cws_word[temp_input_index[k]])
                                temp_cws_label.append(train_label[temp_input_index1[k]])
                                temp_cws_label_.append(train_cws_label[temp_input_index[k]])
                                temp_cws_length.append(train_cws_length[temp_input_index[k]])
                            feed_dict = {}
                            feed_dict[m.input] = np.asarray(temp_cws_word)
                            feed_dict[m.label] = np.asarray(temp_cws_label)
                            feed_dict[m.label_]=np.asarray(temp_cws_label_)
                            feed_dict[m.sent_len] = np.asarray(temp_cws_length)
                            feed_dict[m.is_ner] = 0
                            feed_dict[m.task_label] = np.asarray(task_cws)
                            _, step1, cws_loss= sess.run([train_op1,global_step1,m.cws_loss], feed_dict)
                            if step1 % 500 ==0:
                                temp = "step2 {},cws_loss {}".format(step1, cws_loss)
                                print temp
if __name__ == "__main__":
    tf.app.run()