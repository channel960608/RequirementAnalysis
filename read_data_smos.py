import  xml.dom.minidom

def read_file():
        source = {}
        target = {}
        fileStr_1 = "./SMOS/source_req.xml"
        fileStr_2 = "./SMOS/target_code.xml"
        dom1 = xml.dom.minidom.parse(fileStr_1)
        dom2 = xml.dom.minidom.parse(fileStr_2)
        root1 = dom1.documentElement
        root2 = dom2.documentElement
        artifact_array_source = root1.getElementsByTagName('artifact')
        artifact_array_target = root2.getElementsByTagName('artifact')
        for item in artifact_array_source:                
                art_id = item.getElementsByTagName('id')[0].firstChild.data
                art_content = item.getElementsByTagName('content')[0].firstChild.data
                with open('./SMOS/'+art_content, 'r', encoding='iso8859-1') as f:
                        str_content= f.readlines()
                
                source[art_id] = {'id':art_id, 'title':art_content, 'content':' '.join(str_content)}

        
        for item in artifact_array_target:                
                art_id = item.getElementsByTagName('id')[0].firstChild.data
                art_content = item.getElementsByTagName('content')[0].firstChild.data
                with open('./SMOS/'+art_content, 'r', encoding='iso8859-1') as f:
                        # str_content= f.readlines()
                        content_array = []
                        for line in f:
                                if '*' in line:
                                        content_array.append(line)
                target[art_id] = {'id':art_id, 'title':art_content, 'content':' '.join(content_array)}
        return source, target

def read_link():
        link_dic = []
        fileStr = "./SMOS/answer_req_code.xml"
        dom = xml.dom.minidom.parse(fileStr)
        root = dom.documentElement
        link_array = root.getElementsByTagName('link')
        for item in link_array:
                source = item.getElementsByTagName('source_artifact_id')[0].firstChild.data
                target = item.getElementsByTagName('target_artifact_id')[0].firstChild.data
                link_dic.append((source, target))
        return link_dic
                

        

                        



def load_data():
    source, target = read_file()
    
#     print('pause')
    return source, target

def test():
    source, target = read_file()
    link = read_link()
    print('pause')

if __name__ == "__main__":
#     load_data()
#     print('ok')
    test()
