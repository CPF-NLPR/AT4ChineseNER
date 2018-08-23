import tensorflow as tf
import numpy as np
import Sighan_model
import os
import preprocess_sighan

def get_ner_fmeasure(golden_lists, predict_lists, label_type="BIO"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0, sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        gold_matrix = get_ner_BIO(golden_list)
        pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    accuracy = (right_tag + 0.0) / all_tag
    print "gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num
    return accuracy, precision, recall, f_measure

def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def decode(logits,trans_params,lengths):
    viterbi_sequences=[]
    for logit, length in zip(logits, lengths):
        logit = logit[:length]
        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
        viterbi_sequences += [viterbi_seq]
    return viterbi_sequences

def evaluate(pred_batch,label_batch,length_batch,word_batch):
    result_batch=[]
    for i in range(len(pred_batch)):
        result=[]
        label_batch1=label_batch[i][:length_batch[i]]
        gold=[]
        pred=[]
        for j in range(length_batch[i]):
            gold.append(preprocess_sighan.id_tag[label_batch1[j]])
            pred.append(preprocess_sighan.id_tag[pred_batch[i][j]])
            one_unit=[preprocess_sighan.id2word[word_batch[i][j]],gold[j],pred[j]]
            result.append(" ".join(one_unit))
        result_batch.append(result)
    return result_batch

def compute_f1(results):
    output_file='/home/cpf/ToolKit/NER/data/output_sighan.txt'
    with open(output_file,'w') as f:
        to_write=[]
        for batch in results:
            for sent in batch:
                for word in sent:
                    to_write.append(word+'\n')
                to_write.append('\n')
        f.writelines(to_write)
    f.close()
    gold_label=[]
    gold_label.append([])
    predict_label=[]
    predict_label.append([])

    f=open(output_file,'r')
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            gold_label.append([])
            predict_label.append([])
        else:
            content=content.replace('\n','').replace('\r','').split()
            if len(content)==3:
                gold_label[len(gold_label)-1].append(content[1])
                predict_label[len(predict_label)-1].append(content[2])
            elif len(content)==2:
                gold_label[len(gold_label) - 1].append(content[0])
                predict_label[len(predict_label) - 1].append(content[1])
    f.close()
    if [] in gold_label:
        gold_label.remove([])
    if [] in predict_label:
        predict_label.remove([])
    ACC,P,R,F=get_ner_fmeasure(gold_label,predict_label)
    tempstr = "ACC {},P {},R {},F {}".format(ACC, P, R, F)
    print tempstr

def main(_):
    print 'read word embedding......'
    embedding = np.load('./data/vector.npy')
    print 'read test data......'
    test_word = np.load('./data/test_word.npy')
    test_label = np.load('./data/test_label.npy')
    test_length = np.load('./data/test_length.npy')
    setting = Sighan_model.Setting()
    with tf.Graph().as_default():
        # use GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess=tf.Session(config=config)
        with sess.as_default():
            with tf.variable_scope('ner_model'):
                m = Sighan_model.TransferModel(setting, tf.cast(embedding, tf.float32), adv=True, is_train=False)
                m.multi_task()
            saver=tf.train.Saver()
            for k in range(30500,180000,500):
                saver.restore(sess, './ckpt/lstm+crf-sighan' + '-' + str(k))
                results=[]
                for j in range(len(test_word)//setting.batch_size):
                    word_batch=test_word[j*setting.batch_size:(j+1)*setting.batch_size]
                    length_batch=test_length[j*setting.batch_size:(j+1)*setting.batch_size]
                    label_batch=test_label[j*setting.batch_size:(j+1)*setting.batch_size]
                    feed_dict={}
                    feed_dict[m.input]=word_batch
                    feed_dict[m.sent_len]=length_batch
                    feed_dict[m.is_ner]=1
                    logits, trans_params= sess.run([m.ner_project_logits,m.ner_trans_params],feed_dict)
                    viterbi_sequences=decode(logits,trans_params,length_batch)
                    result_batch=evaluate(viterbi_sequences,label_batch,length_batch,word_batch)
                    results.append(result_batch)
                print 'current_step:%s  The result is:'%(k)
                compute_f1(results)

if __name__ == "__main__":
    tf.app.run()