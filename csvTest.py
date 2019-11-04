import csv

with open('log01.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['batch', 'epoch', 'accuracy'])
   
    for index in range(10, 20):
        writer.writerow([index, index * 2, index * 3])
