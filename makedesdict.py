import pickle
import numpy as np
target_vocab, word_embeddings = pickle.load(file=open("./dataset/ubuntu_data/vocab_and_embeddings.pkl", 'rb'))
ubuntu_cmd_vec = np.load('./dataset/AKdict/command_description.npy').item()
vocab = open('./dataset/AKdict/vocab.txt', 'r').readlines()
id2w = {}
for word in vocab:
    w = word.split('\n')[0].split('\t')
    id2w[int(w[1])] = w[0]

newvector={}
cnt=0
for i in ubuntu_cmd_vec:

    if id2w[i] not in target_vocab:
        #print("pass")
        #cnt+=1
        continue
    t = target_vocab[id2w[i]]
    change=[]
    des_list=ubuntu_cmd_vec[i]
    for k in range(len(des_list)):
        cnt += 1
        if des_list[k]==0:
            break
        if id2w[des_list[k]] not in target_vocab:
            print("pass")
            change.append(target_vocab['unk'])
            continue
        change.append(target_vocab[id2w[des_list[k]]])
    change.extend([0] * (44 - len(change)))
    newvector[t]=change
pickle.dump(newvector, file=open("./dataset/ubuntu_data/command_description.pkl", 'wb'))
    #print(i,id2w[i],target_vocab[id2w[i]])
print(cnt)