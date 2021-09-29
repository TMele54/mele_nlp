def get_lex(path):
    with open(path,'r') as f:
        data = f.readlines()
        f.close()
    return data

J = get_lex('../../data/emotions/lex.csv')

for i in J:
    print i.split(',')