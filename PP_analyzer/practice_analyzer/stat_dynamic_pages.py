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
    classifications = dict()
    for num, ext in enumerate(extList):
        # if num > 2000:
        #     break
        extPath = dirPath+'/'+ext
        extInfo = []
        classification = []
        for home, dirs, files in os.walk(extPath):
            fileSizeList = []
            for filename in files:
                filePath = extPath + '/' + filename
                size = os.path.getsize(filePath)
                fileSizeList.append([filename, size])
            fileSizeList = sorted(fileSizeList, key=lambda i: i[1])
            
            pointer = 0
            for i in range(len(fileSizeList)):
                fileSize = fileSizeList[i]
                if len(classification) <= pointer:
                    classification.append([])
                classification[pointer].append(fileSize[0])

                if i < len(fileSizeList) - 1 and fileSizeList[i + 1][1] - fileSize[1] >= 1024:
                    pointer += 1
        classification = sorted(classification, key=lambda i: -1 * len(i))
        classificationDict = dict()
        for i, group in enumerate(classification):
            classificationDict[i + 1] = group
        classifications[ext] = classificationDict

    save_json('multi_layer_results.json', classifications)
        #         if filename[-5:] == '.html':
        #             try:
        #                 global count
        #                 count+=1
        #                 htmlInfo = { "file_name": filename, "user_input_tages": [] }
        #                 userInputTags=["form","input","button","textarea","select","option",
        #                             "optgroup","fieldset","legend","datalist","output"]
        #                 with open(os.path.join(home, filename), 'r') as f:
        #                     tmp = f.read()
        #                     soup = BeautifulSoup(tmp, 'html.parser')
        #                 # handle the html, extract specific tags
        #                 # currently, we ignore the dynamic generated html tags
        #                 for inputTag in userInputTags:
        #                     collectTag=soup.find_all(inputTag)
        #                     for tagItem in collectTag:
        #                         tmpItem={"tag_name": inputTag, "class": tagItem.get('class'), "id": tagItem.get('id'),
        #                                 "text": tagItem.string, "placeholder":tagItem.get('placeholder'),"raw_html": str(tagItem)}
        #                         htmlInfo["user_input_tages"].append(tmpItem)
        #                 extInfo.append(htmlInfo)    
        #             except Exception as e:
        #                 print(e)
        #                 print(count,'cannot handle file',filename)

        # # has traversed all the html files in an extension    
        # # extTagPath=dirPath+'/'+ext+"_userinput_tags.json"
        # # save_json(extTagPath,extInfo)

def stat_pages():
    content=json.load(open('multi_layer_results.json','r'))
    count = 0
    third_count=0
    for c in content.keys():
        if "2" not in content[c]:
            count += 1
        else:
            # has more than one layer of UI
            idx=2
            while str(idx) in content[c]:
                clickableTags=["button"]
                find_third=False
                home='../raw_data/dynamic_pages_further_process/'+c
                for page in content[c][str(idx)]:
                    with open(os.path.join(home, page), 'r') as f:
                        tmp = f.read()
                        soup = BeautifulSoup(tmp, 'html.parser')
                    # handle the html, extract specific tags
                    # currently, we ignore the dynamic generated html tags
                    
                    collectTag=soup.find_all("button")
                    if len(collectTag)>2:
                        # there is a next page
                        find_third=True
                        break
                if find_third:
                    third_count+=1
                    print('find a thrid in',c)
                    break
                idx+=1

    print(count)
    print(len(content.keys()))
    print(third_count)

if __name__ == "__main__":
    dirPath='../raw_data/dynamic_pages_further_process'
    traverFile(dirPath)
    stat_pages()
