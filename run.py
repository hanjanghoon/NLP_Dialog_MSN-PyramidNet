import time
import argparse
import pickle
from MSN import MSN
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task_dic = {
    'ubuntu':'./dataset/ubuntu_data/',
    'douban':'./dataset/DoubanConversaionCorpus/',
    'alime':'./dataset/E_commerce/'
}
data_batch_size = {
    "ubuntu": 100,
    "douban": 150,
    "alime":  200
}

## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--max_utterances",
                    default=10,
                    type=int,
                    help="The maximum number of utterances.")
parser.add_argument("--max_words",
                    default=50,
                    type=int,
                    help="The maximum number of words for each utterance.")
parser.add_argument("--batch_size",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=100,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
args = parser.parse_args()
args.batch_size = data_batch_size[args.task]
args.save_path += args.task + '.' + MSN.__name__ + ".pt"
args.score_file_path = task_dic[args.task] + args.score_file_path

print(args)
print("Task: ", args.task)


def train_model():
    path = task_dic[args.task]
    X_train_utterances, X_train_responses, y_train = pickle.load(file=open(path+"train.pkl", 'rb'))
    X_dev_utterances, X_dev_responses, y_dev = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    #make_key_r(X_train_responses)
    #make_key_r(X_dev_responses)
    key_r=np.load('./dataset/ubuntu_data/key_r.npy')
    key_mask_r=np.load('./dataset/ubuntu_data/key_mask_r.npy')
    dev_key_r = np.load('./dataset/ubuntu_data/dev_key_r.npy')
    dev_key_mask_r = np.load('./dataset/ubuntu_data/dev_key_mask_r.npy')
    #make_key_r(X_train_responses[500000:1000000],2)
    #idx2sentnece(X_train_utterances, X_train_responses, X_dev_utterances, X_dev_responses, vocab, y_train)
    '''
    k=1000
    X_train_utterances=X_train_utterances[:k]
    X_train_responses=X_train_responses[:k]
    y_train=y_train[:k]
    X_dev_utterances= X_dev_utterances[:k]
    X_dev_responses=X_dev_responses[:k]
    y_dev=y_dev[:k]
    key_r, key_mask_r=key_r[:k], key_mask_r[:k]
    dev_key_r, dev_key_mask_r=dev_key_r[:k],dev_key_mask_r[:k]
    '''


    model = MSN(word_embeddings, args=args)
    model.fit(
        X_train_utterances, X_train_responses, y_train,
        X_dev_utterances, X_dev_responses, y_dev,
        key_r,key_mask_r,dev_key_r,dev_key_mask_r
    )

def get_key(sentence, max_seq_len, max_len):
        """
        get key mask
        :param sentence:
        :param max_len:
        :return:
        """
        ubuntu_cmd_vec = pickle.load(file=open("./dataset/ubuntu_data/command_description.pkl", 'rb'))

        key_mask = np.zeros((max_seq_len))
        keys = np.zeros((max_seq_len, max_len))
        for j, word in enumerate(sentence):
            if int(word) in ubuntu_cmd_vec.keys():
                keys[j] = ubuntu_cmd_vec[int(word)][:max_len]
                key_mask[j] = 1
            else:
                keys[j] = np.zeros((max_len))
        return key_mask, keys

def make_key_r(X_train_responses):
    X_train_shape=np.array(X_train_responses).shape
    key_r = np.zeros([X_train_shape[0],X_train_shape[1],44], np.float32)
    key_mask_r = np.zeros([X_train_shape[0], X_train_shape[1]], np.float32)
    for j, row_r in enumerate(X_train_responses):
        key_mask_r[j], key_r[j] = get_key(row_r, X_train_shape[1], 44)
    np.save('./dataset/ubuntu_data/key_r.npy',key_r)
    np.save('./dataset/ubuntu_data/key_mask_r.npy', key_mask_r)
    '''
    if ver==1:
        pickle.dump([ key_r, key_mask_r], file=open("./dataset/ubuntu_data/key_r&key_mask_r_1.pkl", 'wb'))
    else:
        pickle.dump([key_r, key_mask_r], file=open("./dataset/ubuntu_data/key_r&key_mask_r_2.pkl", 'wb'))
    '''
def idx2sentnece(train_u, train_r, dev_u, dev_r, vocab, y_train):

    # tokenized_texts=tokenized_texts = [bert_tokenizer.tokenize("i am hppy")]
    # print (tokenized_texts[0])

    reverse_vocab = {v: k for k, v in vocab.items()}

    train_bu = []  # 총 백만.
    for i, context in enumerate(train_u):  # context len =10
        context_b = []
        if (i % 100000 == 0):
            print(i)

        for utterance in context:  # utterance max =50
            utterance_b = ""
            for word_idx in utterance:
                if (word_idx == 0): continue
                utterance_b += reverse_vocab[word_idx] + " "
            if (len(utterance_b) == 0):
                continue

            utterance_b = utterance_b[:-1]
            # print(utterance_b)


            # utterance_t+= [0 for i in range(50-len(utterance_t))]#맥스 단어가 50임 빠끄
            context_b.append(utterance_b)
        train_bu.append(context_b)

    train_br = []

    for utterance, y in zip(train_r, y_train):  # utterance max =1문장
        utterance_b = ""
        for word_idx in utterance:
            if (word_idx == 0): continue
            utterance_b += reverse_vocab[word_idx] + " "
        '''
        if (len(utterance_b) == 0):#백만개에서 줄어듬......
            print("response missing!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        '''
        utterance_b = utterance_b[:-1]
        # utterance_t += [0 for i in range(50 - len(utterance_t))]
        train_br.append(utterance_b)
        # print(utterance_t)
    print("end")
    pickle.dump([train_bu, train_br], file=open("sentence/train_ori.pkl", 'wb'))

    dev_bu = []  # 총 백만.
    for context in dev_u:  # context len =10
        context_b = []
        for utterance in context:  # utterance max =50
            utterance_b = ""
            for word_idx in utterance:
                if (word_idx == 0): continue
                utterance_b += reverse_vocab[word_idx] + " "

            if (len(utterance_b) == 0):
                continue
            utterance_b = utterance_b[:-1]
            # print(utterance_b)

            # utterance_t += [0 for i in range(50 - len(utterance_t))]
            context_b.append(utterance_b)
        dev_bu.append(context_b)

    dev_br = []
    for utterance in dev_r:  # utterance max =1문장
        utterance_b = ""
        for word_idx in utterance:
            if (word_idx == 0): continue
            utterance_b += reverse_vocab[word_idx] + " "
        '''
        if (len(utterance_b) == 0):
            continue
        '''
        utterance_b = utterance_b[:-1]
        # utterance_t += [0 for i in range(50 - len(utterance_t))]
        dev_br.append(utterance_b)

    pickle.dump([dev_bu, dev_br], file=open("sentence/dev_ori.pkl", 'wb'))


def test_model():
    path = task_dic[args.task]
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))

    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)

def test_adversarial():
    path = task_dic[args.task]
    vocab, word_embeddings = pickle.load(file=open(path + "vocab_and_embeddings.pkl", 'rb'))
    model = MSN(word_embeddings, args=args)
    model.load_model(args.save_path)
    print("adversarial test set (k=1): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_1.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=2): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_2.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)
    print("adversarial test set (k=3): ")
    X_test_utterances, X_test_responses, y_test = pickle.load(file=open(path+"test_adversarial_k_3.pkl", 'rb'))
    model.evaluate(X_test_utterances, X_test_responses, y_test, is_test=True)


if __name__ == '__main__':
    start = time.time()
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end-start)/60, " min")




