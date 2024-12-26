import sys
import math
import string

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict().fromkeys(string.ascii_uppercase, 0)
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        for line in f:
            line = line.strip()
            line = line.upper()
            words = line.split(" ") # words is an array of words
            for word in words:
                for character in word:
                    if character in X:
                        X[character] = X[character]+1
    return X

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def Q2(X):
    a = X.get('A');
    probs = get_parameter_vectors()
    # print(probs[0][0]) # gives me correct number from english
    english = a * math.log(probs[0][0])
    spanish = a * math.log(probs[1][0])
    print(f'{english:.4f}')
    print(f'{spanish:.4f}')

    # print(f'{0:.4f}'.format(english))
    # print(f'{0:.4f}'.format(spanish))

def Q3(X):
    index = 0
    probs = get_parameter_vectors()
    englishsum = 0
    spanishsum = 0
    
    # Iterates through all 26 lines in the dictionary
    for lines in X.items():
        stats = X.get(chr(index+65)) # use ascii as the key
        englishsum = englishsum + stats*math.log(probs[0][index])
        spanishsum = spanishsum + stats*math.log(probs[1][index])
        index = index+1
    
    #English
    englishreturn = math.log(0.6) + englishsum
    #print(f'{0:.4f}'.format(englishreturn))
    #print(f'{englishreturn:.4f}')

    #Spanish
    spanishreturn = math.log(0.4) + spanishsum
    #print(f'{spanishreturn:.4f}')
    #print(f'{0:.4f}'.format(spanishreturn))
    
    return englishreturn, spanishreturn

def Q4(X):
    
    FEnglish = Q3(X)[0]
    FSpanish = Q3(X)[1]
    
    if FSpanish-FEnglish >= 100:
        return 0
    
    if FSpanish-FEnglish <= -100:
        return 1
    
    return 1/(1+(math.exp(FSpanish-FEnglish)))

def __main__():
    X = shred('letter.txt')
    print("Q1")
    #print(X) #prints out the dictionary
    for character, count in X.items():
        print(character, str(count))
    print("Q2")
    Q2(X)
    print("Q3")
    print(f'{Q3(X)[0]:.4f}')
    print(f'{Q3(X)[1]:.4f}')
    print("Q4")
    print(f'{Q4(X):.4f}')      
    
if __name__ == "__main__":
    __main__()
    