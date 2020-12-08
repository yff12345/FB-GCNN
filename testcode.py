def write_result(file_path, file_name, wtr):
    f = open(file_path + file_name, 'a')
    f.write(wtr)
    f.write('\n')
    f.close()

write_result('C:/Users/HANYIIK/Desktop/', 'k=5、kernel=32、epoch=100.txt', 'hello')