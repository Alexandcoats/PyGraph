import re
import collections

expressions = collections.OrderedDict((
    ('e', r'2.71828'),
    ('pi', r'3.14159'),
    ('sin\(', 'numpy.sin('),
    ('cos\(', 'numpy.cos('),
    ('tan\(', 'numpy.tan('),
    ('sinh\(', r'numpy.sinh('),
    ('cosh\(', r'numpy.cosh('),
    ('tanh\(', r'numpy.tanh('),
    ('sqrt\(', r'numpy.sqrt('),
    ('pow\(', r'numpy.power('),
    ('log\(', r'numpy.log(')))

syntax = r'sin\(|cos\(|tan\(|sinh\(|cosh\(|tanh\(|sqrt\(|pow\(|log\(|e|pi|dgr\(|rad\(|x|y|[0-9]|[-+*/]|[\.,]|[()]|\s'

def replace(string):
    for expr, repl in expressions.items():
        string = re.sub(expr, repl, string)
    return string

def check_valid(string):
    string = re.sub(syntax, '', string)
    return len(string) == 0

def vars(string):
    var = []
    if re.search('x', string) != None:
        var.append('x')
    if re.search('y', string) != None:
        var.append('y')
    return var

#print(check_2d('sin(cos(x))+4*5.1-sinh(y2) + log(pi, e)'))