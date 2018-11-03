from csv import reader, writer

with open('finalmergeok.csv', newline='') as ip, open('Base_Features_count.csv', 'w', newline='') as op:
    item = list(reader(ip))
    item[0].insert(10,'Pages scraped')
    body = item[1:]
    count = 1
    for row in body:
        print("processing row " + str(count))
        count +=1
        features = row[11:]
        try:
            features_int = list(map(lambda x: int(x), features))
            smol = min(features_int)
            if smol > 6:
                print(row)
            if smol == 0:
                item.remove(row)
            features = list(map(lambda x: x - smol, features_int))
            row[11:] = features
            row.insert(10, smol)
        except:
            print(row)
            item.remove(row)
    csv_writer = writer(op)
    csv_writer.writerows(item)

print("All is well")

"""
# cap max count to 20. No significant increase in accuracy. decided not to include this.
with open('training_features_count.csv', newline='') as ip, open('training_features_count_cap.csv', 'w', newline='') as op:
    item = list(reader(ip))
    body = item[1:]
    for row in body:
        features = row[11:]
        features = list(map(lambda x: int(x), features))
        features = list(map(lambda x: min(x, 20), features))
        row[11:] = features
    csv_writer = writer(op)
    csv_writer.writerows(item)

    print("all is well")

"""   


"""

with open('Base_Features.csv', newline='') as ip, open('Base_Features_count.csv', 'w', newline='') as op:
    item = list(reader(ip))
    body = item[1:]
    for row in body:
        features = row[10:]
        try:
            features_int = list(map(lambda x: int(x), features))
            smol = min(features_int)
            if smol > 6:
                print(row)
            features = list(map(lambda x: x - smol, features_int))
            row[10:] = features
        except:
            print(row)
            item.remove(row)
    csv_writer = writer(op)
    csv_writer.writerows(item)

"""