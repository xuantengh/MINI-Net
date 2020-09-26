import os
import shutil
import pdb
def getlist(path,dstName):
    with open(dstName,'w') as f:
        for root,dirs,files in os.walk(path):
            for fe in files:
                size = os.path.getsize(os.path.join(root,fe))
                size = formatSize(size)
                if size>=1:
                    print(fe)
                    f.write(fe+'\n')
                else:
                    # pdb.set_trace()
                    print('remove: '+fe)
                    os.remove(os.path.join(root,fe))
def getDBlist(path,dstName):
    with open(dstName,'w') as f:
        category = []
        for root,dirs,files in os.walk(path):
            for dr in dirs:
                category.append(dr)
        for ctg in category:
            for root,dirs,files in os.walk(os.path.join(path,ctg)):
                for fe in files:
                    f.write(ctg+'/'+fe+'\n')
# 字节bytes转化kb\m\g
def formatSize(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"

    # if kb >= 1024:
    #     M = kb / 1024
    #     if M >= 1024:
    #         G = M / 1024
    #         return  G
    #     else:
    #         return M
    # else:
    return kb

getlist('/home/share/Highlight/orgDataset/instagram/skating','video_list.txt')
