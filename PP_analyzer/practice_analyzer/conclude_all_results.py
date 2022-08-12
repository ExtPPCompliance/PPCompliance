from operator import imod
import os
import json
import csv

chrome_api=["accessibilityFeatures", "browsingData", "commands",
            "contentSettings", "cookies", "declarativeNetRequest",
            "desktopCapture", "devtools",
            "enterprise", "fileBrowserHandler", "history",
            "identity", "instanceID", "permissions", "power", "privacy", "proxy", "runtime",
            "scripting", "storage", "system"]

tag_csv_header=['html_file','tag','id','class','text','placeholder','raw_html']

input_dom=[]

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_csv(path,data,csv_header):
    with open(path,'w') as f:
        writer = csv.writer(f)
        # write a row to the csv file
        header=['ext']+csv_header
        writer.writerow(header)
        writer.writerows(data)

def traverDOM(dir_path):
    res_list=[]
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-19:] == '_dom_operation.json':
                ext_id=item[:-19]

                with open(os.path.join(dir_path, item),'r') as f:
                    file_data=json.load(f)

                print('find the file',ext_id)


def traverAPI(dir_path):
    res_list=[]
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-17:] == '_privacy_api.json':
                ext_id=item[:-17]
                tmp=[ext_id]
                with open(os.path.join(dir_path, item),'r') as f:
                    file_data=json.load(f)
                for api in chrome_api:
                    if api in file_data.keys():
                        # contains the api
                        tmp.append(file_data[api])
                    else:
                        # not contain the api
                        tmp.append(0)
                res_list.append(tmp)        
                print('analysis the ext',item)
    return res_list

def traverTags(dir_path):
    res_list=[]
    for item in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, item)):
            if item[-20:] == '_userinput_tags.json':
                ext_id=item[:-20]

                with open(os.path.join(dir_path, item),'r') as f:
                    file_data=json.load(f)

                for item in file_data:
                    # traverse all html files
                    if len(item["user_input_tages"])!=0:
                        # there is user input tags
                        for tag in item["user_input_tages"]:
                            tmp=[ext_id,item['file_name'],tag['tag_name'],
                                tag['id'],tag['class'],tag['text'],tag['placeholder'],tag['raw_html']]
                            res_list.append(tmp)


                print('handle tags for ext',ext_id)
    return res_list

if __name__ == "__main__":
    dir_path='../raw_data/dynamic_pages'
    api_csv_path='./chrome_api_conclude.csv'
    tag_csv_path='./dynamic_input_tag_conclude.csv'
    
    # res_list=traverDOM(dir_path)

    # res_list=traverAPI(dir_path)
    # save_csv(api_csv_path,res_list,chrome_api)

    res_list=traverTags(dir_path)
    save_csv(tag_csv_path,res_list,tag_csv_header)