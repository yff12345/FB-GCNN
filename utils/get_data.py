import h5py
import numpy as np

npy_path = 'E:/PycharmProjects/EXPERT_GCN/DATASET/'
# npy_path = '/media/data/hanyiik/gcn/dataset/'

mat_path = '/users/hanyiik/documents/代码/mped/'
# mat_path = '/media/data/liyang/MPED/'


"""
【读取 .mat 文件中的 traindata、trainlabel、testdata、testlabel】
:: 输入: 无
:: 输出: list 格式的 30 个人的 traindata(numpy), 
        list 格式的 30 个人的 trainlabel(numpy),
        list 格式的 30 个人的 testdata(numpy), 
        list 格式的 30 个人的 testlabel(numpy)
:: 用法:
        train_dataset_list, train_labelset_list, test_dataset_list, test_labelset_list = get_data()
"""


def get_data():
    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    '''
    :: 功能: 读取一个 .mat 的训练集、测试集
    :: 输入: .mat 文件名
    :: 输出: numpy 格式的 traindata, trainlabel, testdata, testlabel
    '''

    def load_one_mat_file(filename):
        arrays = {}
        f = h5py.File(mat_path + filename, mode='r+')
        for k, v in f.items():
            arrays[k] = np.array(v)

        def resort(arr):
            return np.array([np.transpose(arr[:, item, :]) for item in range(arr.shape[1])])

        get_traindata = resort(arrays['traindata']).astype(np.float32)
        get_trainlabel = np.squeeze(arrays['trainlabel']).astype(np.int)
        get_testdata = resort(arrays['testdata']).astype(np.float32)
        get_testlabel = np.squeeze(arrays['testlabel']).astype(np.int)

        return get_traindata, get_trainlabel, get_testdata, get_testlabel

    '''
    :: 功能: 数据归一化
    :: 输入: numpy 类型的 traindata, testdata(shape=[2520, 62, 5])
    :: 输出: 归一化后的 numpy 类型的 traindata, testdata(shape 不变)
    '''

    def normalize(data):
        minda = np.tile(np.min(data, axis=2).reshape((data.shape[0], data.shape[1], 1)),
                        (1, 1, data.shape[2]))
        maxda = np.tile(np.max(data, axis=2).reshape((data.shape[0], data.shape[1], 1)),
                        (1, 1, data.shape[2]))
        return (data - minda) / (maxda - minda)

    for j in range(30):
        j += 1
        traindata, trainlabel, testdata, testlabel = load_one_mat_file('mped7forgnn{}.mat'.format(j))
        print('load mped7forgnn{}.mat finished!'.format(j))

        traindata_list.append(normalize(traindata))
        trainlabel_list.append(trainlabel)
        testdata_list.append(normalize(testdata))
        testlabel_list.append(testlabel)

    traindataset = traindata_list
    trainlabelset = trainlabel_list
    testdataset = testdata_list
    testlabelset = testlabel_list

    return traindataset, trainlabelset, testdataset, testlabelset


"""
【读取 .npy 文件中的 traindata、trainlabel、testdata、testlabel】
:: 输入: 无
:: 输出: list 格式的 30 个人的 traindata(numpy), 
        list 格式的 30 个人的 trainlabel(numpy)
        list 格式的 30 个人的 testdata(numpy), 
        list 格式的 30 个人的 testlabel(numpy)
:: 用法:
        train_data_list, train_label_list, test_data_list, test_label_list = load_data()
"""


def load_data(flag='large'):
    traindata_list = []
    trainlabel_list = []
    testdata_list = []
    testlabel_list = []

    for k in range(30):
        k += 1

        traindata = np.load(npy_path + 'data_' + flag + '/train_dataset_{}.npy'.format(k))
        trainlabel = np.load(npy_path + 'data_' + flag + '/train_labelset_{}.npy'.format(k))
        testdata = np.load(npy_path + 'data_' + flag + '/test_dataset_{}.npy'.format(k))
        testlabel = np.load(npy_path + 'data_' + flag + '/test_labelset_{}.npy'.format(k))

        traindata_list.append(traindata)
        trainlabel_list.append(trainlabel)
        testdata_list.append(testdata)
        testlabel_list.append(testlabel)

    return traindata_list, trainlabel_list, testdata_list, testlabel_list


# 生成 .npy 文件
def get_npy(flag='large'):
    train_dataset_list, train_labelset_list, test_dataset_list, test_labelset_list = get_data()

    for i in range(30):
        i += 1
        np.save(npy_path + 'data_' + flag + '/train_dataset_{}.npy'.format(i), train_dataset_list[i - 1])
        np.save(npy_path + 'data_' + flag + '/train_labelset_{}.npy'.format(i), train_labelset_list[i - 1])
        np.save(npy_path + 'data_' + flag + '/test_dataset_{}.npy'.format(i), test_dataset_list[i - 1])
        np.save(npy_path + 'data_' + flag + '/test_labelset_{}.npy'.format(i), test_labelset_list[i - 1])
        print('成功保存第{}个人的数据！'.format(i))


# 测试，读取 .npy 文件
def read_npy():
    train_data_list, train_label_list, test_data_list, test_label_list = load_data()

    print(train_data_list[0].shape)
    print(train_label_list[0].shape)
    print(test_data_list[0].shape)
    print(test_label_list[0].shape)
