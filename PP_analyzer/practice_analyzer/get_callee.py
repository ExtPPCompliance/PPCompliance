import os
import json
import csv

privacy_chrome_api_1layer = ["accessibilityFeatures", "browsingData", "commands",
                             "contentSettings", "cookies", "declarativeNetRequest",
                             "desktopCapture", "devtools",
                             "enterprise", "fileBrowserHandler", "history",
                             "identity", "instanceID", "permissions", "power", "privacy", "proxy", "runtime",
                             "scripting", "storage", "system"]
privacy_chrome_api_devtools = ["inspectedWindow", "network", "panels"]
privacy_chrome_api_enterprise = [
    "deviceAttributes", "hardwarePlatform", "networkingAttributes"]

dom_tree_get_api=["getElementsById","getElementsByTagName","getElementsByClassName","body","documentElement","forms"]
dom_tree_creation_api=["createElement","removeChild","appendChild","replaceChild","write"]

def get_AST(file_path):
    AST = []
    with open(file_path, 'r') as f:
        AST = json.load(f)
    return AST


def count_type(AST):
    # calculate the number of types in the ast
    block_num = len(AST['program']['body'])
    empty_state = 0
    print('number of the program blocks: '+str(block_num))
    for item in AST['program']['body']:
        # handle each block respectively
        block_type = item['type']

        if block_type == 'VariableDeclaration':
            block_declaration = item['declarations']
            for decla_item in block_declaration:
                decla_item_id = decla_item['id']

        elif block_type == 'ExpressionStatement':
            block_expression = item['expression']

        elif block_type == 'EmptyStatement':
            # do nothing
            empty_state += 1
        else:
            print('not achieved statement: '+item['type'])


def stat_type(path):
    types = []
    res_names = []
    res = {}
    try:
        with open(path, 'r') as f:
            types = json.load(f)
        for item in types:
            if item['type'] not in res_names:
                res_names.append(item['type'])
                res[item['type']] = 1
            else:
                res[item['type']] += 1

        return res
    except:
        print('cannot get type from file: ', path)
        return []

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def traverseAllCalleeFile(dir_path):
    ext_list=next(os.walk(dir_path))[1]
    for ext in ext_list:
        ext_path=dir_path+'/'+ext
        ext_info={
            "privacy_related_api":0,
            "privacy_related_callee_times":0,
            "privacy_unrelated_callee_times":0,
        }

        for home, dirs, files in os.walk(ext_path):
            for filename in files:
                if filename[-16:] == '_chrome_api.json':
                    try:
                        with open(os.path.join(home, filename), 'r') as f:
                            funcs = json.load(f)
                        for func in funcs:
                            func_names = func["name"].split('.')
                            if len(func_names)>1 and func_names[1] in privacy_chrome_api_1layer:
                                # privacy related api
                                ext_info['privacy_related_callee_times']+=1
                                if func_names[1] in ext_info.keys():
                                    ext_info[func_names[1]]+=1
                                else:
                                    ext_info['privacy_related_api']+=1
                                    ext_info[func_names[1]]=1
                            else:
                                # privacy unlrelated api
                                ext_info['privacy_unrelated_callee_times']+=1
                    except Exception as e:
                        print(e)
                        print('cannot handle file ',filename)
        ext_api_res_path=dir_path+'/'+ext+"_privacy_api.json"
        save_json(ext_api_res_path,ext_info)

def traverseAllDOMTreeFile(dir_path):
    ext_list=next(os.walk(dir_path))[1]
    for ext in ext_list:
        ext_path=dir_path+'/'+ext
        ext_info={"get_element_operation_times":0,"create_element_operation_times":0,"other_operation_times":0,
                    "get_element":[],"create_element":[]}

        for home, dirs, files in os.walk(ext_path):
            for filename in files:
                if filename[-14:] == '_dom_tree.json':
                    try:
                        with open(os.path.join(home, filename), 'r') as f:
                            funcs = json.load(f)
                        for func in funcs:
                            func_names = func["name"].split('.')
                            func["file_name"]=filename[0:-14]
                            if len(func_names)>1:
                                if func_names[1] in dom_tree_get_api:
                                    # get dom element operation
                                    ext_info["get_element_operation_times"]+=1
                                    ext_info["get_element"].append(func)
                                    
                                elif func_names[1] in dom_tree_creation_api:
                                    # create dom element operation
                                    ext_info["create_element_operation_times"]+=1
                                    ext_info["create_element"].append(func)
                                if func_names[1] in ext_info.keys():
                                    ext_info[func_names[1]]+=1
                                else:
                                    ext_info[func_names[1]]=1
                            else :
                                # not target funcion/api
                                ext_info["other_operation_times"]+=1
                                pass
                    except Exception as e:
                        print(e)
                        print('cannot handle file ',filename)
        ext_api_res_path=dir_path+'/'+ext+"_dom_operation.json"
        save_json(ext_api_res_path,ext_info)


def handle_all_callee():
    global all_chrome_api
    callee_path = '../raw_data/process'
    # traverseAllCalleeFile(callee_path)
    traverseAllDOMTreeFile(callee_path)


if __name__ == '__main__':
    handle_all_callee()

