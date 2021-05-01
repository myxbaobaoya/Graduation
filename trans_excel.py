import xlwt
import os
import sys



def txt_xls(filenametrain,filenameresult,xlsname):
    try:
        f = open(filenametrain)
        f2 = open(filenameresult)
        xls = xlwt.Workbook()
        # 生成excel的方法，声明excel
        sheet = xls.add_sheet('sheet' ,cell_overwrite_ok=True)
        x = 0  # 在excel开始写的位置（y）

        while True:  # 循环读取文本里面的内容
            line = f.readline()  # 一行一行的读
            line2 = f2.readline()
            write = 0
            if not line:  # 如果没有内容，则退出循环
                break
            for i in range(len(line.split())):  # \t即tab健分隔
                item = line.split()[i]
                sheet.write(x ,i ,item)  # x单元格经度，i单元格纬度
                write = write + 1
            item2 = line2.split('\n')[0]
            sheet.write(x, write, item2)
            x += 1  # 另起一行
        f.close()
        f2.close()
        xls.save(xlsname)  # 保存为xls文件
    except:
        raise
if __name__ == '__main__':
    filenametrain = 'train_data.txt'
    filenameresult = 'result_data.txt'
    xlsname = 'train_data.xls'
    txt_xls(filenametrain ,filenameresult,xlsname)