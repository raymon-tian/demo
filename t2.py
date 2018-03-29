#!/usr/bin/env python
# -*- coding: utf_8 -*-
# Date: 2016年10月10日
# Author:蔚蓝行

#首先创建一个类cls,这个类中包含一个值为1的类变量clsvar，一个值为2的实例变量insvar,
class cls:
    clsvar = 1
    def __init__(self):
        self.insvar = 2

#创建类的实例ins1和ins2
ins1 = cls()
ins2 = cls()

print cls.__dict__
print ins1.__dict__

#用实例1为类变量重新赋值并打印
print '#'*10
ins1.clsvar = 20
print cls.clsvar     #输出结果为1
print ins1.clsvar    #输出结果为20
print ins2.clsvar    #输出结果为1

#用类名为类变量重新赋值并打印
print '#'*10
cls.clsvar = 10
print cls.clsvar     #输出结果为10
print ins1.clsvar    #输出结果为20
print ins2.clsvar    #输出结果为10

#这次直接给实例1没有在类中定义的变量赋值
print '#'*10
ins1.x = 11
print ins1.x         #输出结果为11

#然后再用类名给类中没有定义的变量赋值
print '#'*10
cls.m = 21
print cls.m          #输出结果为21

#再创建一个实例ins3，然后打印一下ins3的变量
print '#'*10
ins3 = cls()
print ins3.insvar    #输出结果为2
print ins3.clsvar    #输出结果为10
print ins3.m         #输出结果为21
# print ins3.x         #报错AttributeError: cls instance has no attribute 'x'

class Temp(object):


    def __init__(self):
        self.b= 1
        print('f')

    def __setattr__(self, key, value):
        print('ssss\n')
        # self.key = value

class Child(Temp):

    def __init__(self):
        self.a = 2
        print('ch')

# a = Temp()
# a.t1 = 3
# print(a.t1)

b = Child()
print(b.b)