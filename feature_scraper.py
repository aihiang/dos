import re
import copy
import requests
from bs4 import BeautifulSoup
import urllib
import http.client
from urllib.request import urlopen
from urllib.error import URLError
from http.client import HTTPException
from urllib.error import HTTPError
from csv import reader, writer
from multiprocessing import Pool


BASE_WORDS = ['$', 'account', 'accreditation', 'add to cart', 'advert', 'adwords', 'android', 'animation', 'auction', 'automat', 'automobile', 'award', 'b2b', 'b2c', 'bank', 'big', 'blog', 'book', 'breakfast', 'buy', 'car', 'career', 'cart', 'cash', 'casino', 'check out', 'client', 'cloud', 'complete', 'consultancy', 'consultation', 'contact', 'contact form', 'customer', 'cyber', 'data', 'dating', 'deals', 'deliver', 'develop', 'development', 'domain', 'e-store', 'ecommerce', 'emarketing', 'enquiry', 'enrol', 'equipment', 'escort', 'estore', 'facebook', 'firewall', 'form', 'free', 'game', 'get in touch', 'goods', 'hack', 'hangout', 'hostel', 'hosting', 'hotel', 'housing', 'instagram', 'intelligence', 'interactive', 'investor relations', 'iphone', 'job', 'kiosk', 'learn', 'log in', 'log out', 'machine', 'magazine', 'mail', 'mailing', 'mailing list', 'managed', 'marketing', 'marketplace', 'mastercard', 'media', 'member', 'menu', 'message', 'mobile', 'movie', 'multimedia', 'music', 'name', 'network', 'news', 'news alert', 'offers', 'online', 'online store', 'order', 'partners', 'password', 'pay', 'payment', 'paypal', 'phone number', 'platform', 'premium', 'price', 'privacy', 'product', 'projects', 'promotion', 'purchase', 'recognition', 'recruit', 'rent', 'research', 'reserv', 'restaurant', 'return', 'review', 'risk', 'robot', 's$', 'sale', 'service', 'science', 'seo', 'serie', 'server', 'sgd', 'shipping', 'shop', 'shopping bag', 'sign in', 'sign out', 'sign up', 'skype', 'snapchat', 'social media', 'software', 'spyware', 'store', 'stream', 'subject', 'submit', 'subscri', 'supply', 'tablet', 'testimonial', 'ticket', 'transfer', 'travel', 'trial', 'tuition', 'tutor', 'tutorial', 'tv', 'twitter', 'usd', 'username', 'video', 'virtual', 'virus', 'visa', 'vlog', 'vpn', 'web design', 'webdesign', 'what we do', 'youtube']


class CustomException(Exception):
    pass


def string_soup(soup):
    """
    Converts soup object into string
    """
    #here soup return the html of the page
    if soup is None:
        return ''
    else:
        return str(soup).lower()

#atm dont need this?

def write_page(url, str):
    """
    Saves soup text as .txt file
    """
    name = ''.join(x for x in url if x.isalnum())
    dir = 'URLs10/' + name + '.txt'
    file = open(dir, 'w')
    file.write(str)


def open_page(url):
    """
    Takes URL of page and returns its HTML text if found, else returns empty string
    PARAM: URL of company website
    OUTPUT: Soup object, None otherwise
    """
    # lst = []
    try:
        resp = requests.get(url, timeout=120)
        if not 200 <= resp.status_code < 300:
            return None
        main_page = BeautifulSoup(resp.text, 'html5lib')
        return main_page

    except:
        with open('gs_failed_rerun.csv', 'a') as savefile: #a for appending
            csv_writer = writer(savefile)
            csv_writer.writerow([url])
        return None



def get_spawns(url_d, soup):
    """
    Takes Soup object of base URL or None and returns all its subpages
    PARAM: base URL + Soup object of base URL
    OUTPUT: list of base + daughter urls
    """
    url = url_d
    if url[-1] != "/":
        url = url + "/"
    if soup is None:
        return []
    potential_spawns = soup.find_all('a')
    spawns = []
    for spawn in potential_spawns:
        if spawn.get('href') is not None:
            spawns.append(spawn.get('href'))
    lst = [url]
    for l in spawns:
        if 'pdf' in l or 'javascript' in l:
            continue
        if 'http' in l and url_d not in l: #taking those that are https://??
            continue
        if url_d in l:
            spawn = l
        elif re.match(r"/(\w)+", l) and '#' not in l and '@' not in l: #some urls start with /abcdef
            spawn = url + l[1:]
        elif 'http' not in l and '#' not in l and '@' not in l:
            spawn = url + '/' + l
        lst.append(spawn)
    lst = list(set(lst))
    return lst


def sort_spawns(url_d, spawns):
    """
    Extracts and classifies the relevant spawn categories for subsequent processing
    PARAM: base URL + list of spawned URLs
    OUTPUT: dictionary of list of URLs (of interest)
    """
    d = {}
    d['base'] = [url_d]
    d['contact/enquiry'] = []
    d['product'] = []
    d['service'] = []
    if len(spawns) == 0:
        return d
    else:
        for i in spawns:
            if len(d['contact/enquiry']) == 3 and len(d['product']) == 1 and len(d['service']) == 1: #limit to 5 urls in each part of dictionary
                break
            if 'contact' in i or 'enquiry' in i and len(d['contact/enquiry']) < 3: #searching for the word product in url
                d['contact/enquiry'].append(i)
            if 'product' in i and len(d['product']) == 0:
                d['product'].append(i)
            if 'service' in i and len(d['service']) == 0:
                d['service'].append(i)
        return d


def dictionarise(row):
    """
    Takes data from row of base and converts it into dictionary
    PARAM: row of base
    OUTPUT: dictionary with keys of relevant url categories and values as tuple of (sublink, soup obj of sublink, text of soup obj)
    """
    name = row[1]
    comp_url = row[9]
    # name = row[1]
    # comp_url = row[2]
    print(comp_url)
    page = open_page(comp_url)  # get soup object of base url
    if page is None:
        d = {}
        d['base'] = []
        d['contact/enquiry'] = []
        d['product'] = []
        d['service'] = []
        return d
    sublinks = get_spawns(comp_url, page)  # get all sublinks from base url
    relevant_links = sort_spawns(comp_url, sublinks)  # get dictionary of relevant sublinks from base url
    # print(relevant_links)
    try:
        value_store = []
        if len(relevant_links['base']) == 0:
            empty_d = {}
            return empty_d
        for value in relevant_links.values():
            value_store.extend(value)
        value_store = list(set(value_store))  # lists all relevant sublinks 
        #why set and list?
        # print(value_store)
        objects = []  # list of (sublink, soup obj of sublink, text of soup obj) for all sublinks
        for sublink in value_store:
            soup = open_page(sublink)
            if soup is None:
                continue
            string = string_soup(soup)
            objects.append((sublink, soup, string))
            # write_page(sublink, string)
        for object in objects:  # replace urls in relevant_links with (sublink, soup obj of sublink, text of soup obj)
            for value in relevant_links.values():
                for link in value:
                    if type(value) == tuple:
                        continue
                    if link == object[0]:
                        value.append(object)
                        value.remove(link)
        return relevant_links
    except:
        d = {}
        d['base'] = []
        d['contact/enquiry'] = []
        d['product'] = []
        d['service'] = []
        return d

def empty_data_dict():
    """
    Calculates score for keywords of interests
    OUTPUT: dictionary of score for BASE_WORDS
    """
    d = {}
    keywords = BASE_WORDS
    for k in keywords:
        d[k] = 0
    return d


def scorer(t, keywords):
    """
    Takes in keywords and returns dictionary of scores for keywords for a page's text
    PARAM: soup text AND list of keywords
    OUTPUT: dictionary of keyword scores
    """
    d = {}
    for k in keywords:
        d[k] = len(t.split(k)) #if word not found, will return 1. Original sentence, no split
        # words = "This is random text we’re going to split apart"
        # words2 = words.split("text")
        # words2
    return d


def proc_main(url_dict, data_dict):
    """
    Adds score from soup object to score tabulator
    PARAM: dictionary of URLs of interest + dictionary of current score
    OUTPUT: dictionary of score updated with scores from main page
    """
    if 'base' not in url_dict.keys():
        d = {}
        return d
    url_lst = url_dict['base']
    if len(url_lst) == 0:
        return data_dict
    for url in url_lst:
        try:
            # print(url[0])
            #  (sublink, soup obj of sublink, text of soup obj)
            # if less than 3, 1 is missing
            if len(url) < 3:
                continue
            # print(url)
            txt = url[2]
            temp_dict = scorer(txt, BASE_WORDS)
            for key, value in temp_dict.items(): 
                # {'facebook': 10, 'twitter' : 14.... } #data_dict
                data_dict[key] += value
        except Exception as ex:
            print((url, ex.__class__.__name__))
        return data_dict


def proc_prod(url_dict, data_dict):
    """
    Adds score from soup object to score tabulator
    PARAM: dictionary of URLs of interest + dictionary of current score
    OUTPUT: dictionary of score updated with scores from product page
    """
    if 'product' not in url_dict.keys():
        d = {}
        return d
    url_lst = url_dict['product']
    if len(url_lst) == 0:
        return data_dict
    for url in url_lst:
        try:
            # print(url[0])
            if len(url) < 3:
                continue
            # print(url)
            txt = url[2]
            temp_dict = scorer(txt, BASE_WORDS)
            for key, value in temp_dict.items():
                data_dict[key] += value
        except Exception as ex:
            print((url, ex.__class__.__name__))
        return data_dict


def proc_ser(url_dict, data_dict):
    """
    Adds score from soup object to score tabulator
    PARAM: dictionary of URLs of interest + dictionary of current score
    OUTPUT: dictionary of score updated with scores from service page
    """
    if 'service' not in url_dict.keys():
        d = {}
        return d
    url_lst = url_dict['service']
    if len(url_lst) == 0:
        return data_dict
    for url in url_lst:
        try:
            # print(url[0])
            if len(url) < 3:
                continue
            # print(url)
            txt = url[2]
            temp_dict = scorer(txt, BASE_WORDS)
            for key, value in temp_dict.items():
                data_dict[key] += value
        except Exception as ex:
            print((url, ex.__class__.__name__))
        return data_dict


def proc_contact(url_dict, data_dict):
    """
    Adds score from soup object to score tabulator
    PARAM: dictionary of URLs of interest + dictionary of current score
    OUTPUT: dictionary of score updated with scores from contact/enquiry page
    """
    if 'contact/enquiry' not in url_dict.keys():
        d = {}
        return d
    url_lst = url_dict['contact/enquiry']
    if len(url_lst) == 0:
        return data_dict
    for url in url_lst:
        try:
            # print(url[0])
            if len(url) < 3:
                continue
            # print(url)
            txt = url[2]
            temp_dict = scorer(txt, BASE_WORDS)
            for key, value in temp_dict.items():
                data_dict[key] += value
        except Exception as ex:
            print((url, ex.__class__.__name__))
        return data_dict


# #Initial code. dict returns error on keys
# def proc_all(url_dict, data_dict):
#     """
#     Adds score from soup to score tabulator
#     PARAM: dictionary of URLs of interest + dictionary of current score
#     OUTPUT: list of scores according to sorted order of keys (follows order of BASE_WORDS)
#     """
#     lst = []
#     if url_dict is None or url_dict is {} or data_dict is None:
#         return []
#     else:
#         data_dict = proc_main(url_dict, data_dict)
#         data_dict = proc_prod(url_dict, data_dict)
#         data_dict = proc_ser(url_dict, data_dict)
#         data_dict = proc_contact(url_dict, data_dict)
#         lst = []
#         for key in sorted(data_dict.keys()):
#             lst.append(data_dict[key])
#             #empty list now stores new row w num
#         return lst
    
def proc_all(url_dict, data_dict):
    """
    Adds score from soup to score tabulator
    PARAM: dictionary of URLs of interest + dictionary of current score
    OUTPUT: list of scores according to sorted order of keys (follows order of BASE_WORDS)
    """
    if url_dict is None or url_dict is {} or data_dict is None:
        return []
    else:
        data_dict = proc_main(url_dict, data_dict)
        data_dict = proc_prod(url_dict, data_dict)
        data_dict = proc_ser(url_dict, data_dict)
        data_dict = proc_contact(url_dict, data_dict)
        lst = []
        try: 
            for key in sorted(data_dict.keys()):
                lst.append(data_dict[key])
                #empty list now stores new row w num
            return lst
        except:
            return lst
    




def score(row):
    """
    Generates keyword score for an observation
    PARAM: list of entity data
    OUTPUT: extended list of entity data with scores
    """
    scoreboard = empty_data_dict()
    d = dictionarise(row)
    if d is None:
        curr_score = []
    else:
        curr_score = proc_all(d, scoreboard)
    row.extend(curr_score)
    return row


############
# RUN CODE #
############

"""
# Running on macs - parallelized
with open('base_rerun.csv', 'r') as ip, open('base_data_rerun.csv', 'w') as op, Pool(20) as pool:
    base = list(reader(ip))[:20000]
    header = base[0]
    header.extend(BASE_WORDS)
    base[0] = header
    base[1:] = pool.map(score, base[1:])
    print('writing')
    csv_writer = writer(op)
    csv_writer.writerows(base)
    print('written')

"""


# Running on windows, parallelized
if __name__ ==  '__main__':
    with open('try_featuresextraction.csv', 'r', newline='') as ip, open('try_featuresextraction_output.csv', 'w', newline='') as op, Pool(20) as pool:
        base = list(reader(ip))[:20000]
        header = base[0]
        header.extend(BASE_WORDS)
        base[0] = header
        base[1:] = pool.map(score, base[1:])
        print('writing')
        csv_writer = writer(op)
        csv_writer.writerows(base)
        print('written')



"""
# Running on windows, NOT parallelized
with open('try_featuresextraction.csv', 'r', newline='') as ip, open('try_featuresextraction_output.csv', 'w', newline='') as op:
    base = list(reader(ip))[:20000]
    header = base[0]
    header.extend(BASE_WORDS)
    base[0] = header
    body = base[1:]
    count = 1
    for row in body:
        print("processing row " + str(count))
        score(row)
        count +=1
    csv_writer = writer(op)
    csv_writer.writerows(base)
    print('written')

"""