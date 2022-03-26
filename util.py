# 关于__getattr__ 函数的作用
# 如果对对象进行属性查询，没查到失败了，那么就会自动调用类的__getattr__函数
# 如果没有定义这个函数，就会抛出AttributeError

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

