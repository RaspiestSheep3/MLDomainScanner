import sqlite3
import DomainCalculations

numEntries = int(input("Number of entries : "))
dataCSVPath = input("Data CSV : ")
outputPath = input("Domain DB : ")
dictionaryPath = input("Dictoinary path : ")
bigramsPath = input("Bigram path : ")
trigramsPath = input("Trigram path : ")

conn = sqlite3.connect(outputPath)
cursor = conn.cursor()

def GenerateTestingDomains():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS domainsTesting (
            domain STRING PRIMARY KEY,
            shannonEntropy FLOAT,
            vowelConsonantRatio FLOAT,
            longestConsecutiveConsonants FLOAT,
            dictionaryCount FLOAT,
            bigramCount FLOAT,
            trigramCount FLOAT,
            isMalicious INT
        )
    """)
    conn.commit()

    with open(dataCSVPath, "r") as f:
        lines = f.readlines()

    numBenignDomains = 0
    numMaliciousDomains = 0

    i = 0
    while(numBenignDomains < (numEntries // 2) or numMaliciousDomains < (numEntries // 2)):
        if((i + 1) % 100 == 0):
            print(f"{i + 1} / {numEntries} ({(i + 1) * 100/numEntries:.3f}%)")
        
        line = lines[i].strip().split(",")
        domain = line[0]
        isMalicious = int(line[1])

        if(isMalicious not in [0,1]):
            print(f"Warning : {domain} not defnied properly")

        if((isMalicious == 0 and numBenignDomains >= numEntries // 2) or (isMalicious == 1 and numMaliciousDomains >= numEntries // 2)):
            i += 1
            continue
        
        try:
            shannonEntropy = DomainCalculations.ShannonEntropy(domain)
            vowelConsonantRatio = DomainCalculations.VowelConsonantRatio(domain)
            longestConsecutiveConsonants = DomainCalculations.LongestConsecutiveConsonants(domain)
            dictionaryCount = DomainCalculations.DictionaryCount(domain, dictionaryPath)
            bigramsCount = DomainCalculations.NGramDistribution(domain, bigramsPath)
            trigramsCount = DomainCalculations.NGramDistribution(domain, trigramsPath)
            cursor.execute("""
                INSERT INTO domainsTesting (
                    domain,
                    shannonEntropy,
                    vowelConsonantRatio,
                    longestConsecutiveConsonants,
                    dictionaryCount,
                    bigramCount,
                    trigramCount,
                    isMalicious
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                domain,
                shannonEntropy,
                vowelConsonantRatio,
                longestConsecutiveConsonants,
                dictionaryCount,
                bigramsCount,
                trigramsCount,
                isMalicious
            ))
            
            conn.commit()
            
            if(isMalicious == 0):
                numBenignDomains += 1
            else:
                numMaliciousDomains += 1
        except:
            pass
            #print(f"{domain} is repeated")
        finally:
            i += 1

def GenerateValidationDomains():
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS domainsValidation (
                domain STRING PRIMARY KEY,
                shannonEntropy FLOAT,
                vowelConsonantRatio FLOAT,
                longestConsecutiveConsonants FLOAT,
                dictionaryCount FLOAT,
                bigramCount FLOAT,
                trigramCount FLOAT,
                isMalicious INT
            )
        """)
    conn.commit()
    
    with open(dataCSVPath, "r") as f:
        lines = f.readlines()
    
    lines.reverse()

    numBenignDomains = 0
    numMaliciousDomains = 0

    i = 0
    while(numBenignDomains < (numEntries // 2) or numMaliciousDomains < (numEntries // 2)):
        if((i + 1) % 100 == 0):
            print(f"{i + 1} / {numEntries} ({(i + 1) * 100/numEntries:.3f}%)")
        
        line = lines[i].strip().split(",")
        domain = line[0]
        isMalicious = int(line[1])

        if(isMalicious not in [0,1]):
            print(f"Warning : {domain} not defnied properly")

        if((isMalicious == 0 and numBenignDomains >= numEntries // 2) or (isMalicious == 1 and numMaliciousDomains >= numEntries // 2)):
            i += 1
            continue
        
        try:
            shannonEntropy = DomainCalculations.ShannonEntropy(domain)
            vowelConsonantRatio = DomainCalculations.VowelConsonantRatio(domain)
            longestConsecutiveConsonants = DomainCalculations.LongestConsecutiveConsonants(domain)
            dictionaryCount = DomainCalculations.DictionaryCount(domain, dictionaryPath)
            bigramsCount = DomainCalculations.NGramDistribution(domain, bigramsPath)
            trigramsCount = DomainCalculations.NGramDistribution(domain, trigramsPath)
            cursor.execute("""
                INSERT INTO domainsValidation (
                    domain,
                    shannonEntropy,
                    vowelConsonantRatio,
                    longestConsecutiveConsonants,
                    dictionaryCount,
                    bigramCount,
                    trigramCount,
                    isMalicious
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                domain,
                shannonEntropy,
                vowelConsonantRatio,
                longestConsecutiveConsonants,
                dictionaryCount,
                bigramsCount,
                trigramsCount,
                isMalicious
            ))
            
            conn.commit()
            
            if(isMalicious == 0):
                numBenignDomains += 1
            else:
                numMaliciousDomains += 1
        except:
            pass
            #print(f"{domain} is repeated")
        finally:
            i += 1

GenerateValidationDomains()