
def log_write(outputs,point):
    log = []
    for x0,y0,x1,y1,id,in outputs:
        x0 = x0 + point[0]
        x1 = x1 + point[0]
        w = x1 - x0
        h = y1 - y0
        log_tmp = [id,x0,y0,x1,y1]
        log.append(log_tmp)
    return log

def log_save(save_path,content):
    with open(save_path,'a+') as f:
        f.write(content)
        f.write('\n')
    f.close()