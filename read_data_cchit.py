import  xml.dom.minidom

def read_file(filename):
        source = {}
        fileStr_1 = "./CCHIT/" + filename + ".xml"
        fileStr_2 = "./CCHIT/" + filename + "2.xml"
        dom1 = xml.dom.minidom.parse(fileStr_1)
        dom2 = xml.dom.minidom.parse(fileStr_2)
        root1 = dom1.documentElement
        root2 = dom2.documentElement
        artifact_array_title = root1.getElementsByTagName('artifact')
        artifact_array_content = root2.getElementsByTagName('artifact')
        for item in artifact_array_title:                
                art_id = item.getElementsByTagName('art_id')[0].firstChild.data
                art_title = item.getElementsByTagName('art_title')[0].firstChild.data
                artifact = {}
                artifact['title'] = art_title
                artifact['id'] = art_id
                source[art_id] = artifact
        
        for item in artifact_array_content:                
                art_id = item.getElementsByTagName('id')[0].firstChild.data
                art_content = item.getElementsByTagName('content')[0].firstChild.data
                artifact = source[art_id]
                artifact['content'] = art_content
        return source

def read_link():
        link_dic = []
        with open('./CCHIT/answer.txt', 'r') as f:
            for line in f:
                link = line.split(",")
                pre = link[0]
                aft = link[1].replace("\n","")
                link_dic.append((pre, aft))
        return link_dic

                        



def load_data():
    source = read_file("source")
    target = read_file("target")
#     print('pause')
    return source, target

def test():
    link_dic = read_link()
    print(link_dic)

if __name__ == "__main__":
    load_data()
    print('ok')
#     test()
