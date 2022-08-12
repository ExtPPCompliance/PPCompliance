import os
import json
import csv
from bs4 import BeautifulSoup

count=0
def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def traverFile(dirPath):
    extList = next(os.walk(dirPath))[1]
    for ext in extList:
        extPath = dirPath+'/'+ext
        extInfo = []

        for home, dirs, files in os.walk(extPath):
            for filename in files:
                if filename[-5:] == '.html':
                    try:
                        global count
                        count+=1
                        htmlInfo = { "file_name": filename, "user_input_tages": [] }
                        userInputTags=["form","input","button","textarea","select","option",
                                    "optgroup","fieldset","legend","datalist","output"]
                        with open(os.path.join(home, filename), 'r') as f:
                            tmp = f.read()
                            soup = BeautifulSoup(tmp, 'html.parser')
                        # handle the html, extract specific tags
                        # currently, we ignore the dynamic generated html tags
                        for inputTag in userInputTags:
                            collectTag=soup.find_all(inputTag)
                            for tagItem in collectTag:
                                tmpItem={"tag_name": inputTag, "class": tagItem.get('class'), "id": tagItem.get('id'),
                                        "text": tagItem.string, "placeholder":tagItem.get('placeholder'),"raw_html": str(tagItem)}
                                htmlInfo["user_input_tages"].append(tmpItem)
                        extInfo.append(htmlInfo)    
                    except Exception as e:
                        print(e)
                        print(count,'cannot handle file',filename)
        # has traversed all the html files in an extension    
        extTagPath=dirPath+'/'+ext+"_userinput_tags.json"
        save_json(extTagPath,extInfo)

if __name__ == "__main__":
    dirPath='../raw_data/dynamic_pages'
    traverFile(dirPath)
