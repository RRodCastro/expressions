from math import ceil

M = [
    'E014M', 'E014V', 'E016M', 'E016V', 'E018M', 'E018V',
    'E026M', 'E026V', 'E027M', 'E027V', 'E030M', 'E030V',
    'E031M', 'E031V', 'E033M', 'E033V', 'E034M', 'E034V',
    'E038M', 'E038V', 'E045M', 'E045V', 'E046M', 'E046V',
    'E054M', 'E054V', 'E060M', 'E060V', 'E062M', 'E062V',
    'E065M', 'E065V', 'E067M', 'E067V']

data = open('./during-answers.csv', 'r')

frecuency = {}
for interview in M:
    if("V" in interview):
        continue
    frecuency[interview] = {'L': 0, 'N': 0, 'R': 0}
frecuency.keys()

for i in data:
    test = i.split(",")
    if(test[2] == '0\n' or test[2] == '0'):
        frecuency[i[0:5]]['L'] += 1
    elif(test[2] == '2\n' or test[2] == '2'):
        frecuency[i[0:5]]['N'] += 1
data.close()
for i in frecuency:
    if(frecuency[i]['L'] > 0):
        frecuency[i]['R'] = ceil(frecuency[i]['N']/frecuency[i]['L'])

print(frecuency)
data = open('./during-answers.csv', 'r')

writer = open('cleaned-da.csv',  'a')
temp = ''
count = 0
for line in data:
    current = line[0:5]
    # print(current)
    if(current != temp):
        count = 0
        temp = current
    test = line.split(",")
    line = test[0] + "," + test[1] + "," + test[2]
    if(test[2] == '2\n' or test[2] == '2'):
        count += 1
        if(frecuency[current]['R'] != 0 and count % frecuency[current]['R'] == 0):
            writer.write(line)
    else:
        writer.write(line)

writer.close()
