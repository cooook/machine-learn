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
    tmp_attributes = json.loads(
        '{"attributes":{"argumentTypes":[{"typeIdentifier":"t_uint256","typeString":"uint256"}],"isConstant":false,'
        '"isLValue":false,"isPure":false,"lValueRequested":false,"member_name":"send","referencedDeclaration":null,'
        '"type":"function (uint256) returns (bool)"}}')[
        'attributes']
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


def find_judge_in_father(node):
    while node:
        if node.name == 'IfStatement':
            return True
        node = node.father

    return False


def CheckReturnValue(node):
    UnaryOperation_node = Node()
    UnaryOperation_node.parseAttributes(json.loads(
        '{"argumentTypes":null,"isConstant":false,"isLValue":false,"isPure":false,"lValueRequested":false,"operator":"!","prefix":true,"type":"bool"}'))
    UnaryOperation_node.name = "UnaryOperation"
    father_node = node.father
    UnaryOperation_node.children.append(father_node)

    IfStatement_node = Node()
    IfStatement_node.parseAttributes(json.loads('{"falseBody":null}'))
    IfStatement_node.name = 'IfStatement'
    IfStatement_node.children.append(UnaryOperation_node)
    UnaryOperation_node.father = IfStatement_node

    parseAst(json.loads('{"children":[{"children":[],"id":25,"name":"Throw","src":"233:5:0"}],"id":26,"name":"Block"}'),
             IfStatement_node)

    # throw 可能存在问题， 这个版本的throw 直接向IfStatement_node增加一个节点

    pass


def MatchUncheckReturnValue(node):  # 对低级函数调用进行匹配
    if node.member_name == 'send':
        if find_judge_in_father(node) == False:
            CheckReturnValue(node)
    for ch in node.children:
        MatchUncheckReturnValue(ch)

'''
对于 二目运算 a (x)= b
使用SafeMath库中的函数来替代
SafeMath's node -> ls = 之前表达式的node
rs = 运算表达式node
对于 a x b
直接转换， 不需要改变其他的东西
'''
def MatchOperator(opt):
    if opt == '+=':
        return ['add', 1]
    if opt == '-=':
        return ['sub', 1]
    if opt == '*=':
        return ['mul', 1]
    if opt == '+':
        return ['add', 0]
    if opt == '-':
        return ['sub', 0]
    if opt == '*':
        return ['mul', 0]
    return None

def Add_Node(father, NewNode, node):
    for x in range(len(father)):
        if father.children[x] == node:
            father.children[x] = NewNode
    NewNode.father = father

def UsingSafeMathLibrary(node, mode):
    # mode[1] 为 1 代表是 'x=' 需要增加等号 和 左值
    FirstNode = Node()
    SubNode = Node()
    SafeMathNode = Node()
    FirstNode.parseAttributes('{"attributes":{"argumentTypes":null,"isConstant":false,"isLValue":false,"isPure":false,"isStructConstructorCall":false,"lValueRequested":false,"names":[null],"type":"uint256","type_conversion":false}}')
    SubNode.parseAttributes('{"attributes":{"argumentTypes":[{"typeIdentifier":"t_uint256","typeString":"uint256"},{"typeIdentifier":"t_uint256","typeString":"uint256"}],"isConstant":false,"isLValue":false,"isPure":false,"lValueRequested":false,"member_name":"{0}","referencedDeclaration":68,"type":"function (uint256,uint256) pure returns (uint256)"}}'.format(mode[0]))
    SafeMathNode.parseAttributes('{"attributes":{"argumentTypes":null,"overloadedDeclarations":[null],"referencedDeclaration":93,"type":"type(library SafeMath)","value":"SafeMath"}}')
    FirstNode.name = 'FunctionCall'
    SubNode.name = 'MemberAccess'
    SafeMathNode.name = 'Identifier'


    SafeMathNode.father = SubNode
    SubNode.children.append(SafeMathNode)

    FirstNode.children.append(SubNode)
    for x in node.children:
        FirstNode.children.append(x)
        x.father = FirstNode

    # for x in range(len(node.father)):
    #     if node.father.children[x] == node:
    #         node.father.children[x] = FirstNode
    # FirstNode.father = node.father

    if not mode[1]:
        Add_Node(node.father, FirstNode, node)
        return 

    EqualNode = Node()
    EqualNode.parseAttributes('{"attributes":{"argumentTypes":null,"isConstant":false,"isLValue":false,"isPure":false,"lValueRequested":false,"operator":"=","type":"uint256"}}')
    EqualNode.name = 'Assignment'

    NewBalanceNode = Node()
    NewBalanceNode.parseAttributes(node.children[0].attributes)
    NewBalanceNode.name = 'IndexAccess'
    
    NewBalanceNode.father = EqualNode
    EqualNode.children.append(NewBalanceNode)
    
    FirstNode.father = EqualNode
    EqualNode.children.append(FirstNode)

    Add_Node(node.father, EqualNode, node)
    return 
    # node1 -> children = [SubNode[SafeMathNode], balances[sender[msg]], _amount]
    # SubNode -> father.children = node1


def MatchUnsafeMath(node):
    if 'operator' in node.attributes:
        opt = node.attributes['operator']
        mode = MatchOperator(opt)
        if mode:
            UsingSafeMathLibrary(node, mode)
    for ch in node.children:
        MatchUnsafeMath(ch)

def MatchAccessControl(node):
    pass 
'''
1. isConstructor => Judge Function 
2. dangrous fucntion call such as pay eth must verify their idx  such as developer, owner
    2.1 If have Constructor 
            if have owner
                add it
            else 
                add owner in Constructor and add Judge
        if don't have Constructor 
            Suggest user to add Constructor
Our Work: 
'''

if __name__ == '__main__':
    pass