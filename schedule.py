# from time import time, sleep
# while True:
#     if 60 - time() % 60 == 0:
#         print(0, time(), 60 - time() % 60)
#     if 60 - time() % 60 == 1:
#         print(1, time(), 60 - time() % 60)        

import random
import time

start = time.clock()
print(start)
while time.clock() - start < 3:
    random_number = random.randint(0,100)

print(random_number)