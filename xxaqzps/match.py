import json

file_name = './2.ast'
with open(file_name) as ast_file:
    data = ast_file.read()
    ast_data = json.loads(data)


class Node:
    # nodeType
    # children
    # attribute
    # src
    # beginPoint
    # endPoint
    # father
    def __init__(self, nodeType=None, father=None):
        self.nodeType = nodeType
        self.attributes = None
        self.children = []
        self.father = father
        self.beginPoint = 0
        self.endPoint = 0
        self.member_name = None
        self.name = None

    def parseSrc(self, src):
        src = list(map(int, src.split(':')))
        self.beginPoint = src[0]
        self.endPoint = src[1] + src[0]

    def parseAttributes(self, attributes):
        self.attributes = attributes
        self.nodeType = attributes['type']
        if 'member_name' in attributes:
            self.member_name = attributes['member_name']


def parseAst(ast_node, father):
    node = Node(father=father)
    if father:
        father.children.append(node)
    if 'attributes' in ast_node:
        node.parseAttributes(ast_node['attributes'])
    if 'src' in ast_node:
        node.parseSrc(ast_node['src'])
    if 'name' in ast_node:
        node.name = ast_node['name']
    if 'children' in ast_node:
        for x in ast_node['children']:
            parseAst(x, node)


def find_value_in_father(node):
    flag = False
    while node.nodeType != 'function () payable returns (bool)':
        if node.member_name == 'value':
            flag = True
        node = node.father
    if flag:
        return node
    return None


'''
将call.value()修改为send()
新建一个send_node节点, 父亲为ast上first_node的父亲
要修改node 即 call的所有children 的父亲和first_node 的孩子， 注意孩子的顺序. 
'''
def change_calldotvalue(first_node, node):
    tmp_attributes = json.loads('{"attributes":{"argumentTypes":null,"isConstant":false,"isLValue":false,"isPure":false,"lValueRequested":false,"member_name":"call","referencedDeclaration":null,"type":"function () payable returns (bool)"}}')['attributes']
    send_node = Node(nodeType='function (uint256) returns (bool)', father=first_node.father)
    send_node.parseAttributes(tmp_attributes)
    # 改儿子的父亲
    send_node.children = node.children
    for x in send_node.children:
        x.father = send_node
    # 改父亲的儿子
    # 不改变顺序的基础上直接改
    for x in range(len(send_node.father.children)):
        if send_node.father.children[x] == first_node:
            send_node.father.children[x] = send_node
            break

    return send_node



def MatchReentrancy(node):
    if node.member_name == 'call':
        first_node = find_value_in_father(node)
        if first_node:
            send_node = change_calldotvalue(first_node, node)
            MatchReentrancy(send_node)
    for ch in node.children:
        MatchReentrancy(ch)


def MatchUncheckReturnValue(node):  # 对低级函数调用进行匹配
    pass
