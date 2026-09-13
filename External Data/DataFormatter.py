def FormatDomains():
    #I dont want the TLD servers like .org, .co.uk because its not in the training data
    testingFile = input("Testing file path : ")
    outputFile = input("Output file path : ")
    with open(testingFile, "r") as fIn:
        with open(outputFile, "w") as fOut:
            lines = fIn.readlines()
            
            for line in lines:
                lineSplit = line.split(",")
                fOut.write(lineSplit[0] + "," + lineSplit[-1])

def FormatWordList():
    #I dont want any words which are 1 or 2 letters because there'll be too many detections on random strings
    inFile = input("Raw file : ")
    outFile = input("Out file : ")
    
    lengths = []
    
    with open(inFile, "r") as fIn:
        with open(outFile, "w") as fOut:
            lines = fIn.readlines()
            
            for line in lines:
                line = line.strip()
                
                if(len(line) >= 3):
                    fOut.write(line + "\n")
        
                lengths.append(len(line))
    
    print(f"Max length : {max(lengths)}")

def FormatNGrams():
    #Manually removed all bigrams with <1m entries, trigrams 500k
    #Removing entry amojunts
    fInPath = input("NGrams raw : ")
    fOutPath = input("NGrams out : ")
    
    with open(fInPath, "r") as fIn:
        with open(fOutPath, "w") as fOut:
            lines = fIn.readlines()
            
            for line in lines:
                line = line.strip()
                
                fOut.write(line.split(",")[0] + "\n") 

FormatNGrams()