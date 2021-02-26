# Importing Image from PIL package  
from typing import SupportsIndex
from PIL import Image
import matplotlib.pyplot as plt 
  
# creating a image object 
im = Image.open(r"..\service-two\saved_img.jpg") 


pixels = list(im.getdata())
width, height = im.size
pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]

one = 0
two = 0
three = 0
four = 0
five = 0
six = 0
seven = 0
eith = 0
nine = 0

# (64, 30, 106), (63, 29, 105), (62, 30, 105), (61, 29, 104)

for i in range(len(pixels)  ):
    listToStr = ' '.join([str(elem) for elem in pixels[i]])
    aux = listToStr.replace(",", "").replace(")", "").replace("(", "")

    numbers = [int(i) for i in aux.split() if i.isdigit()] 

    for i in range(len(numbers)):
        if str(numbers[1]).startswith('1'):
            one += 1
        if str(numbers[1]).startswith('2'):
            two += 1
        if str(numbers[1]).startswith('3'):
            three += 1
        if str(numbers[1]).startswith('4'):
            four += 1
        if str(numbers[1]).startswith('5'):
            five += 1
        if str(numbers[1]).startswith('6'):
            six += 1
        if str(numbers[1]).startswith('7'):
            seven += 1
        if str(numbers[1]).startswith('8'):
            eith += 1
        if str(numbers[1]).startswith('9'):
            nine += 1

total = one + two + three + four + five + six + seven + eith + nine

print(total)
print(one)
print(two)

onep = (one*100)/total
twop = (two*100)/total
threep = (three*100)/total
fourp = (four*100)/total
fivep = (five*100)/total
sixp = (six*100)/total
sevenp = (seven*100)/total
eithp = (eith*100)/total
ninep = (nine*100)/total

print("1: "+str(onep))
print("2: "+str(twop))
print("3: "+str(threep))
print("4: "+str(fourp))
print("5: "+str(fivep))
print("6: "+str(sixp))
print("7: "+str(sevenp))    
print("8: "+str(eithp))
print("9: "+str(ninep))
print("total: "+str(onep + twop + threep + fourp + fivep + sixp + sevenp + eithp + ninep))

y = [onep, twop, threep, fourp, fivep, sixp, sevenp, eithp, ninep]
x  = [1,2,3,4,5,6,7,8,9]    

plt.style.use('ggplot')


x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='green')
plt.xlabel("First Digit of Pixels")
plt.ylabel("Amount of Pixels (%)")
plt.title("Benfords Law for Image Pixels")

plt.xticks(x_pos, x)

plt.show()