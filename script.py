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


def fibonacci(n):
    a,b=0,1
    while a<n:
        print(a)
        a,b=b,a+b#a = b,b=a+b
    print()

fibonacci(10)

