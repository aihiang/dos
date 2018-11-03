from csv import reader, writer
import re

def is_building(s):
    return 'hub' in s or 'plaza' in s or 'shop' in s or 'tower' in s or 'warehouse' in s or 'mall' in s or 'house' in s or 'building' in s or 'bldg' in s or 'box' in s or 'square' in s or 'resort' in s or 'gallery' in s or 'mart' in s

def get_postal(string):
    m = re.findall('\d{6}', string)
    if len(list(m)) >= 1:
        return m[-1]
    else:
        return 0

def get_unit(string):
    m = re.search('\d+-\d+', string)
    if m is not None:
        return str(m.group(0))
    else:
        return ''

def clean_add(address):
    """
    Note: this removes unit numbers and postal codes from address. Please get out unit numbers before cleaning add
    """
    add_=address.strip().replace("'", " ").replace('""', '"').replace('"', '').lower()
    add__=re.sub("\s +", " ", add_)
    add__=re.sub("s\(\d{6}\)", "~", add_)
    add__= re.sub('\d+-\d+|\d{6}|#', '~', add__)
    add = re.split(', |~|,', add__)
    lst=[]
    for i in add:
        i=i.strip() #removes excessive empty spaces
        lst.append(i)
    return list(filter(lambda x: x!= '', lst))


with open('6. sgtech_to_check_linkages.csv', newline='') as uncleaned, open('sgtech_add_clean.csv', 'w', newline='') as data_out:
    l = list(reader(uncleaned, delimiter=','))
    header = l[0]
    header.extend(['REG_BLKHSE_NO', 'REG_STR_NM', 'REG_BLDG_NM', 'REG_LEVEL_NO', 'REG_UNIT_NO', 'REG_POSTAL_CODE'])
    clean = [header]
    print(clean)    
    for row in l[1:]:
        i = 0
        while (row[1][i] is '"' and row[1][-i] is not '"') or (row[1][i] is "'" and row[1][-i] is not "'"):
            row[1] += '"'
            i += 1
        postal = get_postal(row[1])
        room = get_unit(row[1]).split('-')
        if len(room) == 2:
            level, unit = room
        else:
            level, unit = 0, 0
            #empty, leave it as 0
        a = clean_add(row[1])
        # a is now a list of address parts, without postal code and unit no.
        possible_roads = []
        buildings = []
        for i in a:
            r = re.search('\d+\w*\s+\w+', i)
            if r is not None:
                possible_roads.append(i)
            if is_building(i):
                buildings.append(i)
        if len(possible_roads) >= 1:
            road = possible_roads[0]
            if re.match('\d+\w*\s+', road) is not None:
                m = re.match('\d+\w*\s+', road)
                house = m.group(0).strip()
                street = road[len(house):].strip()
            else:
                house = 0
                street = road
        else:
            road = 0
            house = 0
            street = 0
        if len(buildings) >= 1:
            building = buildings[0]
        else:
            building = 0
        row.extend([house, street, building, level, unit, postal])
        clean.append(row)
    csv_writer = writer(data_out)
    csv_writer.writerows(clean)
    print('CLEAN!')