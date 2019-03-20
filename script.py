import numpy as np

# class Test:
# words = ["hello","xia","hangli11111"]
# for w in words[:]:
#     if len(w)>6:
#         words.insert(0,w)
# for w in words:
#     print(w)
#
# for n in range(2,10):
#     for x in range(2,n):
#         if n%x == 0:
#             print (n ,"equals",x,'*',n//x)
#             break
#     else:
#         print(n, 'is a prime number')

# def fibonacci(n):
#     a,b=0,1
#     while a<n:
#         print(a)
#         a,b=b,a+b#a = b,b=a+b
#     print()
#
# fibonacci(10)

# Default Argument Values
# def ask_ok(prompt, retries=4, reminder="Please try again!"):
#     while True:
#         ok = input(prompt)
#         if ok in ('y', 'ye', 'yes'):
#             return True
#         if ok in ('no', 'n', 'noop'):
#             return False
#         retries = retries - 1
#         if retries < 0:
#             raise ValueError("retries too many times")
#         print(reminder)

# ask_ok('Do you really want to quit?')
# ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')

    # def __repr__():
    #     return f'{self.__class__.__name__}({self._name}, birthyear: {self.birthyear})'

def say_hello(self):
    return f"Hey there!"

def simple_decorator(function):
    return function

say_hello = simple_decorator(say_hello)



# print(say_hello)

def goodbye(function):
    def wrapper():
        original_output = function()
        new_output = original_output + f" Goodbye, have a good day!"
        return new_output #再返回new_output

    return wrapper #先返回wrapper


@goodbye
def say_hello():
    return f"Hey there!"

# goodbye = goodbye(say_hello)

# print(say_hello())

#Wrapping functions that take input arguments

def goodbye(function):
    def wrapper(*args, **k1wargs):
        original_output = function(*args, **k1wargs)
        new_output = original_output + f" Goodbye, have a good day!"
        return new_output
    return wrapper

def say_hello():
    return f"Hey there!"

@goodbye
def say_words(person, words):
    return f"{person} says: {words}"


a = np.zeros(5)
a

print(say_words("Cleon", "Hey Flynn!"))