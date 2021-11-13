import json

import requests
from bs4 import BeautifulSoup

class Graph:
    def __init__(self,Nodes,is_directed=False):
        self.nodes=Nodes
        self.adj_list={}
        self.is_directed=is_directed

        for node in self.nodes:
            self.adj_list[node]=[]

    def add_edge(self,v,e):
        self.adj_list[v].append(e)
        if not self.is_directed:
            self.adj_list[e].append(v)

    def degree_vertex(self,node):
        degree=len(self.adj_list[node])
        return degree

    # def print_adj(self):
    #     for node in self.nodes:
    #         print(node,":",self.adj_list[node])

def findInOutDegree(adjList, n):
    _in = {}
    out = {}

    for i in adjList:

        List = list(adjList[i])

        # Out degree for ith vertex will be the count
        # of direct paths from i to other vertices
        out[i] = len(List)
        for j in range(0, len(List)):
            # Every vertex that has
            # an incoming edge from i
            if List[j] not in _in:
                _in[List[j]] = 0
            _in[List[j]] += 1

    # print("Vertex\tIn\tOut")
    # for k in out:
    #     print('out degree of ', k , ' is ', out[k], ' and  in degree is ', _in[k])
    return (_in, out)
def isConnected(n1, n2, res, in_dict, out_dict ):
    if n2 in res[n1]:
        return round(1/ out_dict[n1], 3)
    return 0


def JobPosting(url, res):

    if url in res:
        return
    elif len(res)==500:
        return
    else:
        res[url] = set()

    # r = requests.get(url, allow_redirects=False)
    session = requests.Session()
    session.max_redirects = 100
    r = session.get(url)
    soup = BeautifulSoup(r.content, 'html5lib')

    links = soup.findAll('a')

    for row in links:
        link = row.get('href')
        if link and ('http' in link[0:5] or 'https' in link[0:5]):
            res[url].add(link)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",url ," ",len(res[url]))
    stack = list(res[url]) + []
    # print()
    # print()
    # # print("*************************************************************************************************")
    # print()
    # print()
    # print(res)

    while len(stack) != 0:
        url_next = stack.pop(0)
        JobPosting(url_next, res)
    return

def __del__(self):
    state = res
    # print('................dest................', state)
    state = json.dumps(state, indent=4)
    with open("state.json", "w") as file:
        file.write(state)
    file.close()

if __name__ == '__main__':
    res = {}
    JobPosting('https://www.msit.ac.in/', res)
    # JobPosting('https://www.thehindu.com/', res)
    # print("******************************************************************************************************8")
    nodes = res.keys()
    graph = Graph(nodes, is_directed=True)
    for k, v in res.items():
        for a in v:
            graph.add_edge(k, a)
    # graph.print_adj()
    in_dict, out_dict = findInOutDegree(res, len(nodes))
    # print(graph.degree_vertex("B"))

    final = []
    for k in res.keys():
        l = []
        for p in res.keys():
            l.append(isConnected(p, k, res, in_dict, out_dict))
        final.append(l)

    # print('\n'.join([''.join(['{:10}'.format(item) for item in row])
    #                  for row in final]))
    import contextlib

    file_path = 'randomfile.txt'
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            print('\n'.join([''.join(['{:10}'.format(item) for item in row])
                             for row in final]))