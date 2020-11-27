
import json


### acc


### F1
def getF1(pres, labels):

    count_all = len(labels)

    tp = 0
    fp = 0
    fn = 0
    
    for i in range(count_all):
        #print(pres[i][0], labels[i])
        if pres[i][0] > 0.5:
            if labels[i] == 0:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] != 1:
                fn += 1


    # print(pres.shape, labels.shape)
    # print(pres[0])
    # print(labels[0])

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*recall*precision / (recall+precision)
    return precision, recall, f1_score



### mAP
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:  #VOC在2010之后换了评价方法，所以决定是否用07年的
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):  #  07年的采用11个点平分recall来计算
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 取一个recall阈值之后最大的precision
            ap = ap + p / 11.  # 将11个precision加和平均
    else:  # 这里是用2010年后的方法，取所有不同的recall对应的点处的精度值做平均，不再是固定的11个点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))  #recall和precision前后分别加了一个值，因为recall最后是1，所以
        mpre = np.concatenate(([0.], prec, [0.])) # 右边加了1，precision加的是0

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])  #从后往前，排除之前局部增加的precison情况

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]  # 这里巧妙的错位，返回刚好TP的位置，
                                                                                      # 可以看后面辅助的例子

        # and sum (\Delta recall) * prec   用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# 计算每个类别对应的AP，mAP是所有类别AP的平均值
def voc_eval(result_json_path,classname,use_07_metric=False):
    
    with open(result_json_path,'r') as f:
        result_json = json.loads(f.readlines()[0])  

    result_json = sorted(result_json, key=lambda x:x['score'], reverse=True)
    #print(result_json[:10])

    count = len(result_json)
    tp = np.zeros(count) # 用于标记每个检测结果是tp还是fp
    fp = np.zeros(count)
    npos = 0

    for i,item in enumerate(result_json):
        #print(item)

        if classname in item['path']:
            npos += 1
        
        if item['category'] == classname:
            if classname in item['path']:
                tp[i] = 1
            else:
                fp[i] = 1


    # compute precision recall
    fp = np.cumsum(fp) # 累加函数np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

# 计算每个类别对应的AP，mAP是所有类别AP的平均值
def voc_test(result_json_path,classname, label_json_path):
    
    with open(result_json_path,'r') as f:
        result_json = json.loads(f.readlines()[0])  

    result_json = sorted(result_json, key=lambda x:x['score'], reverse=True)
    #print(result_json[:10])

    with open(label_json_path,'r') as f:
        label_json = json.loads(f.readlines()[0])  
    label_imgs = label_json[classname]#testA_v3_clean
    npos = len(label_imgs)
    print("len label:", npos)


    count = len(result_json)
    tp = np.zeros(count) # 用于标记每个检测结果是tp还是fp
    fp = np.zeros(count)
    

    for i,item in enumerate(result_json):
        #print(item)
        
        if item['category'] == classname:
            if os.path.basename(item['image_name']) in label_imgs:
                tp[i] = 1
            else:
                fp[i] = 1


    # compute precision recall
    fp = np.cumsum(fp) # 累加函数np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap


# def get_val_mAP(result_json_path, classname_list):
#     AP_list = []
#     for classname in classname_list:

#         rec, prec, ap = voc_eval(result_json_path, classname)
#         #print(classname, ap)
#         AP_list.append(ap)
#     return np.mean(AP_list)


def get_test_mAP(result_json_path, classname_list, label_json_path):
    AP_list = []
    for classname in classname_list:

        rec, prec, ap = voc_test(result_json_path, classname, label_json_path)
        print("AP %s: %f" % (classname,ap))
        AP_list.append(ap)
    return np.mean(AP_list)



# 计算每个类别对应的AP，mAP是所有类别AP的平均值
def vocOnline(pres, labels, cate_id):
    # print(pres, labels, cate_id)
    # b

    count = len(pres)
    tp = np.zeros(count) # 用于标记每个检测结果是tp还是fp
    fp = np.zeros(count)
    npos = 0

    for i,item in enumerate(pres):
        #print(item)
        if labels[i]==cate_id:
            npos += 1
        
        if item>0.33:
            if labels[i]==cate_id:
                tp[i] = 1
            else:
                fp[i] = 1

    #print(npos)
    # compute precision recall
    fp = np.cumsum(fp) # 累加函数np.cumsum([1, 2, 3, 4]) -> [1, 3, 6, 10]
    tp = np.cumsum(tp)
    rec = tp / (float(npos)+0.000001)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap

def getValmAP(pres, labels):

    class_name = ['calling', 'normal', 'smoking','smoking_calling']

    AP_list = []
    print()
    for idx in range(len(class_name)):
        rec, prec, ap = vocOnline(pres[:,idx], labels, idx)
        #print(class_name[idx], ap)
        AP_list.append(ap)
    return np.mean(AP_list)
