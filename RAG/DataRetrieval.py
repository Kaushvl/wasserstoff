import requests
import re
import json
requests_session = requests.Session()
from VDatabase import PreProcessData

def extractWebsiteName(url):
    # Ensure the URL starts with http:// or https://
    if not re.match('(?:http|ftp|https)://', url):
        url = 'https://' + url
    
    # Regex pattern to match the protocol (http or https) and the domain name up to .com/.org/.net etc.
    pattern = r'(http[s]?://[\w\-\.]+?\.[\w]+)'
    
    # Search the URL for matches
    match = re.search(pattern, url)
    
    # Return the matched domain name if found, otherwise None
    return match.group(1) if match else None

def fetchUrlData(strUrl:str):

    WebInfoFilePath = "RAG\WebPageCount.json"

    domainUrl = extractWebsiteName(strUrl)

    if not domainUrl :
        raise ValueError("Not a proper domain url")
    
    urlConst = "wp-json/wp/v2/posts"

    urlToHit = domainUrl + '/' + urlConst
    # response = requests.get(url)
    posts = requests_session.get(urlToHit, timeout = 20)

    jsonData = posts.json()

    liIds = []
    siteListData = []
    proJson = {}
    for data in jsonData:
        liIds.append(data['id'])
        proJson[data['id']] = data

    with open(WebInfoFilePath) as file:
        AllWebData = json.load(file)
    
    if domainUrl in AllWebData["WebData"].keys():
        listPageIds = AllWebData["WebData"][domainUrl]

        if len(listPageIds) < len(liIds):
            AllWebData["WebData"][domainUrl] = liIds
            lsProcessPageData = []
            
            for pageId in liIds:
                pageData = {}
                pageData['id'] = pageId
                pageData['title'] = proJson[pageId]['title']['rendered']
                pageData['content'] = proJson[pageId]['content']['plain']
                lsProcessPageData.append(pageData) 
            bOutput = PreProcessData(lsProcessPageData)
            print(bOutput)  

    json_object = json.dumps(AllWebData, indent=4)
 
    # Writing to WbePageCounts
    with open(WebInfoFilePath, "w") as outfile:
        outfile.write(json_object)

    return 


print(fetchUrlData("time.com"))