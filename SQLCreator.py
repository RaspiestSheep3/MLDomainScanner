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

cursor.execute("""
    CREATE TABLE IF NOT EXISTS domains (
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

for i in range(numEntries):
    if((i + 1) % 100 == 0):
        print(f"{i + 1} / {numEntries} ({(i + 1) * 100/numEntries}%)")
    
    line = lines[i].strip().split(",")
    domain = line[0]
    isMalicious = int(line[1])

    shannonEntropy = DomainCalculations.ShannonEntropy(domain)
    vowelConsonantRatio = DomainCalculations.VowelConsonantRatio(domain)
    longestConsecutiveConsonants = DomainCalculations.LongestConsecutiveConsonants(domain)
    dictionaryCount = DomainCalculations.DictionaryCount(domain, dictionaryPath)
    bigramsCount = DomainCalculations.NGramDistribution(domain, bigramsPath)
    trigramsCount = DomainCalculations.NGramDistribution(domain, trigramsPath)
    try:
        cursor.execute("""
            INSERT INTO domains (
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
    except:
        pass
        #print(f"{domain} is repeated")