import random
from numpy import *   # 导入numpy的库函数
import gensim
import time

model = gensim.models.Word2Vec.load("C:/Users/USER/Desktop/Project/multi-classification/CNN/word4.model")

fr = open('C:/Users/USER/Desktop/Project/multi-classification/MulLabelTrain3.ss', encoding='utf-8')

label_and_text = [line.split('\t\t') for line in fr]
text = [line[1].replace("<sssss>", "") for line in label_and_text]
label = [line[0] for line in label_and_text]    # 文本标签
sentences = [line.split(' ') for line in text]  # 文本内容


stopword = [ "a","able","about","above","abst","accordance","according","accordingly","across","act",
             "actually","added","adj","affected","affecting","affects","after","afterwards","again",
             "against","ah","all","almost","alone","along","already","also","although","always","am",
             "among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone",
             "anything","anyway","anyways","anywhere","apparently","approximately","are","aren","arent",
             "arise","around","as","aside","ask","asking","at","auth","available","away","awfully","b",
             "back","be","became","because","become","becomes","becoming","been","before","beforehand",
             "begin","beginning","beginnings","begins","behind","being","believe","below","beside","besides",
             "between","beyond","biol","both","brief","briefly","but","by","c","ca","came","can","cannot","can't",
             "cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains",
             "could","couldnt","d","date","did","didn't","different","do","does","doesn't","doing","done","don't",
             "down","downwards","due","during","e","each","ed","edu","effect","eg","eight","eighty","either","else",
             "elsewhere","end","ending","enough","especially","","et","et-al","etc","even","ever","every","everybody",
             "everyone","everything","everywhere","ex","except","f","far","few","ff","fifth","first","five","fix","followed",
             "following","follows","for","former","formerly","forth","found","four","from","further","furthermore","g","gave",
             "get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","had","happens",
             "hardly","has","hasn't","have","haven't","having","he","hed","hence","her","here","hereafter","hereby","herein",
             "heres","hereupon","hers","herself","hes","hi","hid","him","himself","his","hither","home","how","howbeit","however",
             "hundred","i","id","ie","if","i'll","im","immediate","immediately","importance","important","in","inc","indeed","index",
             "information","instead","into","invention","inward","is","isn't","it","itd","it'll","its","itself","i've","j","just","k",
             "keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least",
             "less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","m","made","mainly",
             "make","makes","many","may","maybe","me","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml",
             "more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","na","name","namely","nay","nd","near",
             "nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","no","nobody",
             "non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","now","nowhere","o","obtain","obtained",
             "obviously","of","off","often","oh","ok","okay","old","omitted","on","once","one","ones","only","onto","or","ord","other","others",
             "otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","particular",
             "particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly",
             "present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather",
             "rd","re","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research",
             "respectively","resulted","resulting","results","right","run","s","said","same","saw","say","saying","says","sec","section","see",
             "seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","she","shed","she'll","shes",
             "should","shouldn't","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six",
             "slightly","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon",
             "sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","such",
             "sufficiently","suggest","sup","sure","t","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that",
             "that'll","thats","that've","the","their","theirs","them","themselves","then","thence","there","thereafter","thereby","thered",
             "therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","these","they","theyd","they'll",
             "theyre","they've","think","this","those","thou","though","thoughh","thousand","throug","through","throughout","thru","thus","til",
             "tip","","to","together","too","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","under",
             "unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used","useful","usefully","usefulness","uses",
             "using","usually","v","value","various","'ve","very","via","viz","vol","vols","vs","w","want","wants","was","wasn't","way","we","wed",
             "welcome","we'll","went","were","weren't","we've","what","whatever","what'll","whats","when","whence","whenever","where","whereafter",
             "whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","while","whim","whither","who","whod","whoever","whole",
             "who'll","whom","whomever","whos","whose","why","widely","willing","wish","with","within","without","won't","words","world","would",
             "wouldn't","www","x","y","yes","yet","you","youd","you'll","your","youre","yours","yourself","yourselves","you've"]
punctation = [",", ".", "``", "''", "-", "'", "~", ";", ":"]

# 删除停用词和标点符号
for line in sentences:
    for item in line:
        if item in stopword:
            line.remove(item)
        if item in punctation:
            line.remove(item)


for i, item in enumerate(label):
    if item == "MID":
        label[i] = 1
    elif item == "LOW":
        label[i] = 0
    else:
        label[i] = 2

label = mat(label)  # 标签矩阵
# init
filter1 = []    # 过滤器1
filter2 = []    # 过滤器2
filter3 = []    # 过滤器3
filter4 = []    # 过滤器4

filter_sum = 4
each_filter_num = 1


filter1 = mat(random.rand(1, 300))
filter2 = mat(random.rand(2, 300))
filter3 = mat(random.rand(3, 300))
filter4 = mat(random.rand(4, 300))

g_filter1 = mat(random.rand(1, 300))
g_filter2 = mat(random.rand(2, 300))
g_filter3 = mat(random.rand(3, 300))
g_filter4 = mat(random.rand(4, 300))


k_max = 5                                           # k-max pooling
w = mat(random.rand(filter_sum*k_max, 3))           # 有20个feature map
w = w - 0.5

g_w = mat(random.rand(filter_sum*k_max, 3))           # 有20个feature map
g_w = g_w - 0.5
max_pool = mat(zeros((filter_sum*k_max, 1)))
max_pool_pos = mat(zeros((filter_sum * k_max, 1)))    # 用于记录卷积池值对应的被卷积数据的位置
landa = 0.001                                         # 用于正则化
y = mat(zeros((3, 1)))                                # 输出结果 列向量
max_loop = 1                                         # 迭代的最大次数
omg = 0.001                                           # 学习率
m = len(sentences)                                    # 训练数据大小
print(m)
flag = [0, 0, 0, 0]
max_right = 0


# 卷积运算
def conv(mat1, mat2, mode, max_pool_index):
    a = []
    if mode == 1:
        for index2, item2 in enumerate(mat1):
            a.append(multiply(item2, mat2).sum())
    elif mode == 2:
        for k in range(mat1.shape[0]):
            if k >= 1:
                a.append(multiply(mat1[k-1:k+1], mat2).sum())
    elif mode == 3:
        for k in range(mat1.shape[0]):
            if k >= 2:
                a.append(multiply(mat1[k - 2:k+1], mat2).sum())
    elif mode == 4:
        for k in range(mat1.shape[0]):
            if k >= 3:
                a.append(multiply(mat1[k - 3:k+1], mat2).sum())
    b = argsort(a)                                                       # 获得是排序的索引
    cov_k_max = []
    if flag[mode-1] == 0:
        for k in range(k_max):
            cov_k_max.append(0)
        for k in range(len(b)):
            cov_k_max[k] = a[k]
            max_pool_pos[max_pool_index+k] = k
    else:
        b_tem = sort(b[len(b) - k_max:len(b)])
        b[len(b) - k_max:len(b)] = b_tem
        for k in range(len(b)-k_max, len(b)):
            cov_k_max.append(a[b[k]])                                    # 获得k_max
            max_pool_pos[max_pool_index+k-len(b)+k_max] = b[k]
    cov_k_max = mat(cov_k_max)
    cov_k_max = cov_k_max.T
    return cov_k_max


# 随机梯度下降
for i in range(max_loop):
    start = time.clock()
    cnt = 0
    for line in sentences:
        flag = [1, 0, 0, 0]
        if len(line) < 5:
            flag[0] = 0
        if len(line) >= 8:
            flag[3] = 1
            flag[2] = 1
            flag[1] = 1
        elif len(line) >= 7:
            flag[2] = 1
            flag[1] = 1
        elif len(line) >= 6:
            flag[1] = 1
        text_label = mat(zeros((3, 1)))
        text_label[label[0, cnt]] = 1                              # 每个文本对应标签
        cnt += 1
        models = []                                                # 每个句子对应的词向量组成的矩阵
        for item in line:                                          # 将训练数据中心化
            tem = model[item]
            models.append(tem)                                     # 获得词向量矩阵
        models = mat(models)                                       # 转换为矩阵
# 卷积层至最大化层，共4个feature map
        max_pool[0:k_max, 0] = conv(models, filter1, 1, 0)
        max_pool[k_max:2*k_max, 0] = conv(models, filter2, 2, k_max)
        max_pool[2*k_max:3*k_max, 0] = conv(models, filter3, 3, 2*k_max)
        max_pool[3*k_max:4*k_max, 0] = conv(models, filter4, 4, 3*k_max)
# 最大化层至输出层

# ReLu激活函数
        for j in range(filter_sum * k_max):
            if max_pool[j, 0] < 0:
                max_pool[j, 0] = 0
# 输出值
        y = (max_pool.T * w).T
        y = exp(y)
        y = y / y.sum()
# 权值更新
        for j in range(0, each_filter_num * k_max):
            if max_pool[j, 0] > 0:
                filter1 = filter1 + omg * (w[j] * (text_label - y))[0, 0] * models[int(max_pool_pos[j, 0])]
        for j in range(each_filter_num*k_max, 2 * each_filter_num*k_max):
            if max_pool[j, 0] > 0:
                filter2 = filter2 + omg * (w[j] * (text_label - y))[0, 0] * models[int(max_pool_pos[j, 0]):int(max_pool_pos[j,0])+2]
        for j in range(2*each_filter_num * k_max, 3*each_filter_num * k_max):
            if max_pool[j, 0] > 0:
                filter3 = filter3 + omg * (w[j] * (text_label - y))[0, 0] * models[int(max_pool_pos[j, 0]):int(max_pool_pos[j,0])+3]
        for j in range(3*each_filter_num * k_max, 4*each_filter_num * k_max):
            if max_pool[j, 0] > 0:
                filter4 = filter4 + omg * (w[j] * (text_label - y))[0, 0] * models[int(max_pool_pos[j, 0]):int(max_pool_pos[j,0])+4]
        w = w + omg * max_pool * (text_label - y).T - landa * w

# 验正口袋，记录结果最好的权值矩阵
    right = 0
    for index, line in enumerate(sentences):
        flag = [1, 0, 0, 0]
        if len(line) < 5:
            flag[0] = 0
        if len(line) >= 8:
            flag[3] = 1
            flag[2] = 1
            flag[1] = 1
        elif len(line) >= 7:
            flag[2] = 1
            flag[1] = 1
        elif len(line) >= 6:
            flag[1] = 1
        models = []  # 每个句子对应的词向量组成的矩阵
        for item in line:
            tem = model[item]
            models.append(tem)  # 获得词向量矩阵

        models = mat(models)  # 转换为矩阵
        # 卷积层至最大化层，共120个feature map
        max_pool[0:k_max, 0] = conv(models, filter1, 1, 0)
        max_pool[k_max:2 * k_max, 0] = conv(models, filter2, 2, k_max)
        max_pool[2 * k_max:3 * k_max, 0] = conv(models, filter3, 3, 2 * k_max)
        max_pool[3 * k_max:4 * k_max, 0] = conv(models, filter4, 4, 3 * k_max)
        # 最大化层至输出层
        y = (max_pool.T * w).T
        y = exp(y)
        y = y / y.sum()
        max_pos = argmax(y)
        if max_pos == label[0, index]:
            right += 1
    print(right)
    if right > max_right:
        max_right = right
        g_filter1 = filter1
        g_filter2 = filter2
        g_filter3 = filter3
        g_filter4 = filter4
        g_w = w
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)


right = 0
for index, line in enumerate(sentences):
    flag = [1, 0, 0, 0]
    if len(line) < 5:
        flag[0] = 0
    if len(line) >= 8:
        flag[3] = 1
        flag[2] = 1
        flag[1] = 1
    elif len(line) >= 7:
        flag[2] = 1
        flag[1] = 1
    elif len(line) >= 6:
        flag[1] = 1
    models = []  # 每个句子对应的词向量组成的矩阵
    for item in line:
        tem = model[item]
        models.append(tem)  # 获得词向量矩阵

    models = mat(models)  # 转换为矩阵
    # 卷积层至最大化层，共120个feature map
    max_pool[0:k_max, 0] = conv(models, filter1, 1, 0)
    max_pool[k_max:2 * k_max, 0] = conv(models, filter2, 2, k_max)
    max_pool[2 * k_max:3 * k_max, 0] = conv(models, filter3, 3, 2 * k_max)
    max_pool[3 * k_max:4 * k_max, 0] = conv(models, filter4, 4, 3 * k_max)
    # 最大化层至输出层
    y = (max_pool.T * w).T
    y = exp(y)
    y = y / y.sum()
    max_pos = argmax(y)
    if max_pos == label[0, index]:
        print(max_pos)
        right += 1
print(right)


# 真实数据预测
test = open('C:/Users/USER/Desktop/Project/multi-classification/MulLabelTest.ss', encoding='utf-8')
output = open('C:/Users/USER/Desktop/Project/multi-classification/CNN/result.csv', 'w+')
label_and_text = [line.split('\t\t') for line in test]
text = [line[1].replace("<sssss>", "") for line in label_and_text]
sentences = [line.split(' ') for line in text]  # 文本内容
for index, line in enumerate(sentences):
    flag = [1, 0, 0, 0]
    if len(line) < 5:
        flag[0] = 0
    if len(line) > 8:
        flag[3] = 1
        flag[2] = 1
        flag[1] = 1
    elif len(line) > 7:
        flag[2] = 1
        flag[1] = 1
    elif len(line) > 6:
        flag[1] = 1
    models = []  # 每个句子对应的词向量组成的矩阵
    sur = 0
    omit = 0
    for item in line:
        try:
            if item == "!":
                sur += 1
            if item == "...":
                omit += 1
            tem = model[item]
            tem = model[item]
            models.append(tem)  # 获得词向量矩阵
        except KeyError:
            continue
    models = mat(models)  # 转换为矩阵
    # 卷积层至最大化层，共120个feature map
    max_pool[0:k_max, 0] = conv(models, filter1, 1, 0)
    max_pool[k_max:2 * k_max, 0] = conv(models, filter2, 2, k_max)
    max_pool[2 * k_max:3 * k_max, 0] = conv(models, filter3, 3, 2*k_max)
    max_pool[3 * k_max:4 * k_max, 0] = conv(models, filter4, 4, 3*k_max)
    # 最大化层至输出层
    y = (max_pool.T * w).T
    y = exp(y)
    y = y / y.sum()
    max_pos = argmax(y)             # 预测结果
    if max_pos == 0:
        output.write("LOW\n")
    elif max_pos == 1:
        if sur >= 3:
            output.write("HIG\n")
        elif omit >= 3:
            output.write("LOW\n")
        else:
            output.write("MID\n")

    else:
        output.write("HIG\n")