from math import log2

def ShannonEntropy(input : str) -> float:
    #Shannon Entropy describes how "random" a string is - repeating characters makes the string less random
    
    processedData = set()
    length = len(input)
    
    out = 0
    for char in input:
        if(char in processedData):
            continue
        
        out += (input.count(char) / length) * log2(input.count(char) / length)
        processedData.add(char)
        
    return -out / log2(40) #It seems fair to assume we wont have more than 40 different characters

def VowelConsonantRatio(input : str) -> float:
    #Ratio = 1 => Only vowels, ratio = 0 => only consonants
    # Characters like _, - ignored for this
    
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    vowelCount = 0
    consonantCount = 0
    
    for char in input:
        if(char in vowels):
            vowelCount += 1
        elif(char in consonants):
            consonantCount += 1
    
    return vowelCount / (vowelCount + consonantCount)

def LongestConsecutiveConsonants(input : str) -> float:
    #Returns the longest continuous set of consonants
    consonants = "bcdfghjklmnpqrstvwxyz"
    
    lengths = []
    count = 0
    for char in input:
        if(char in consonants):
            count += 1
        else:
            lengths.append(count)
            count = 0
    
    if(len(lengths) == 0):
        lengths.append(count)
            
    return min(max(lengths), 20) / 20 #Capping length @ 20 => 0 - 1

def DictionaryCount(input : str, wordListFilePath : str) -> float:
    #Returns how many of the words exist in the top 10k words of English
    with open(wordListFilePath, "r") as f:
        lines = f.readlines()
    
    count = 0
    for line in lines:
        line = line.strip()
        
        if(line in input):
            count += 1
    
    return min(count, 10) / 10 #Normalised between 0 and 1 - It seems unlikely to include more than 10 words

def NGramDistribution(input : str, ngramTableFilePath : str) -> float:
    count = 0
    
    with open(ngramTableFilePath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        count += input.count(line)
    
    return count / 40 #Normalised between 0 and 1

def TotalLength(input : str) -> float:
    return len(input) / 70

def DigitPercentage(input : str) -> float:
    digits = ["1", "2", "3", "4", "5", "7", "8", "9", "0"]
    digitsSum = sum(list([input.count(digit) for digit in digits]))
    return digitsSum / len(input)

def LongestConsecutiveDigits(input : str) -> float:
    digits = "1234567890"
        
    lengths = []
    count = 0
    for char in input:
        if(char in digits):
            count += 1
        else:
            lengths.append(count)
            count = 0
    
    if(len(lengths) == 0):
        lengths.append(count)
            
    return min(max(lengths), 20) / 20 #Capping length @ 20 => 0 - 1

def LetterDigitSymbolTransitionCount(input : str) -> float:
    letters = "qwertyuiopasdfghjklzxcvbnm"
    digits = "1234567890"
    symbols = "-._"
    
    transitionCount = 0
    
    for i in range(len(input) - 1):
        char = input[i]
        nextChar = input[i+1]
        
        if(char in letters and nextChar not in letters) or (char in digits and nextChar not in digits) or (char in symbols and nextChar not in symbols):
            transitionCount += 1
    
    return transitionCount / 70