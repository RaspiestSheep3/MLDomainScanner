import random
import sqlite3
import DomainCalculations
from plyer import notification

numEntriesTraining = int(input("Number of entries for training: "))
numEntriesValidation = int(input("Number of entries for validation: "))
dataCSVPath = input("Data CSV : ")
outputPath = input("Domain DB : ")
dictionaryPath = input("Dictoinary path : ")
bigramsPath = input("Bigram path : ")
trigramsPath = input("Trigram path : ")

conn = sqlite3.connect(outputPath)
cursor = conn.cursor()

def GenerateTestingDomains():
    cursor.execute("""
            DROP TABLE IF EXISTS domainsTesting
        """)
        
    conn.commit()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS domainsTesting (
            domain STRING PRIMARY KEY,
            shannonEntropy FLOAT,
            vowelConsonantRatio FLOAT,
            longestConsecutiveConsonants FLOAT,
            dictionaryCount FLOAT,
            bigramCount FLOAT,
            trigramCount FLOAT,
            totalLength FLOAT,
            digitPercentage FLOAT,
            longestConsecutiveDigits FLOAT,
            letterDigitSymbolTransitionCount FLOAT,
            indexOfCoincidence FLOAT,
            isMalicious INT,
            CONSTRAINT entry UNIQUE (
                shannonEntropy,
                vowelConsonantRatio,
                longestConsecutiveConsonants,
                dictionaryCount,
                bigramCount,
                trigramCount,
                totalLength,
                digitPercentage,
                longestConsecutiveDigits,
                letterDigitSymbolTransitionCount,
                indexOfCoincidence
            )
        )
    """)
    conn.commit()

    with open(dataCSVPath, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)

    numBenignDomains = 0
    numMaliciousDomains = 0

    i = 0
    while(numBenignDomains < (numEntriesTraining // 2) or numMaliciousDomains < (numEntriesTraining // 2)):
        if((numBenignDomains + numMaliciousDomains) % 1000 == 0 and ((numBenignDomains + numMaliciousDomains) != 0)):
            conn.commit()
            print(f"{numBenignDomains + numMaliciousDomains} / {numEntriesTraining} ({(numBenignDomains + numMaliciousDomains) * 100/numEntriesTraining:.3f}%)")
        
        line = lines[i].strip().split(",")
        domain = line[0]
        isMalicious = int(line[1])

        if(isMalicious not in [0,1]):
            print(f"Warning : {domain} not defnied properly")

        if((isMalicious == 0 and numBenignDomains >= numEntriesTraining // 2) or (isMalicious == 1 and numMaliciousDomains >= numEntriesTraining // 2)):
            i += 1
            continue
        
        try:
            shannonEntropy = DomainCalculations.ShannonEntropy(domain)
            vowelConsonantRatio = DomainCalculations.VowelConsonantRatio(domain)
            longestConsecutiveConsonants = DomainCalculations.LongestConsecutiveConsonants(domain)
            dictionaryCount = DomainCalculations.DictionaryCount(domain, dictionaryPath)
            bigramsCount = DomainCalculations.NGramDistribution(domain, bigramsPath)
            trigramsCount = DomainCalculations.NGramDistribution(domain, trigramsPath)
            totalLength = DomainCalculations.TotalLength(domain)
            digitPercentage = DomainCalculations.DigitPercentage(domain)
            longestConsecutiveDigits = DomainCalculations.LongestConsecutiveDigits(domain)
            letterDigitSymbolTransitionCount = DomainCalculations.LetterDigitSymbolTransitionCount(domain)
            indexOfCoincidence = DomainCalculations.IndexOfCoincidence(domain)
            
            cursor.execute("""
                INSERT OR IGNORE INTO domainsTesting (
                    domain,
                    shannonEntropy,
                    vowelConsonantRatio,
                    longestConsecutiveConsonants,
                    dictionaryCount,
                    bigramCount,
                    trigramCount,
                    totalLength,
                    digitPercentage,
                    longestConsecutiveDigits,
                    letterDigitSymbolTransitionCount,
                    indexOfCoincidence,
                    isMalicious
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                domain,
                shannonEntropy,
                vowelConsonantRatio,
                longestConsecutiveConsonants,
                dictionaryCount,
                bigramsCount,
                trigramsCount,
                totalLength,
                digitPercentage,
                longestConsecutiveDigits,
                letterDigitSymbolTransitionCount,
                indexOfCoincidence,
                isMalicious
            ))
            
            if cursor.rowcount > 0:
                if isMalicious == 0:
                    numBenignDomains += 1
                else:
                    numMaliciousDomains += 1
            
        except sqlite3.IntegrityError:
            print(f"Warning : Integrity error")
        except ZeroDivisionError:
            print(f"Warning : Division by 0")
        except Exception as e:
            print(f"Warning : {e}")
            assert 1 == 2
            #print(f"{domain} is repeated")
        finally:
            i += 1
    
    conn.commit()

def GenerateValidationDomains():
    cursor.execute("""
        DROP TABLE IF EXISTS domainsValidation
    """)
    
    conn.commit()
    
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS domainsValidation (
                domain STRING PRIMARY KEY,
                shannonEntropy FLOAT,
                vowelConsonantRatio FLOAT,
                longestConsecutiveConsonants FLOAT,
                dictionaryCount FLOAT,
                bigramCount FLOAT,
                trigramCount FLOAT,
                totalLength FLOAT,
                digitPercentage FLOAT,
                longestConsecutiveDigits FLOAT,
                letterDigitSymbolTransitionCount FLOAT,
                indexOfCoincidence FLOAT,
                isMalicious INT,
                CONSTRAINT entry UNIQUE (
                    shannonEntropy,
                    vowelConsonantRatio,
                    longestConsecutiveConsonants,
                    dictionaryCount,
                    bigramCount,
                    trigramCount,
                    totalLength,
                    digitPercentage,
                    longestConsecutiveDigits,
                    letterDigitSymbolTransitionCount,
                    indexOfCoincidence
                )
            )
        """)
    conn.commit()
    
    with open(dataCSVPath, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
    
    numBenignDomains = 0
    numMaliciousDomains = 0

    i = 0
    while(numBenignDomains < (numEntriesValidation // 2) or numMaliciousDomains < (numEntriesValidation // 2)):
        if((numBenignDomains + numMaliciousDomains) % 1000 == 0):
            conn.commit()
            print(f"{numBenignDomains + numMaliciousDomains} / {numEntriesValidation} ({(numBenignDomains + numMaliciousDomains) * 100/numEntriesValidation:.3f}%)")
        
        line = lines[i].strip().split(",")
        domain = line[0]
        isMalicious = int(line[1])

        if(isMalicious not in [0,1]):
            print(f"Warning : {domain} not defnied properly")

        if((isMalicious == 0 and numBenignDomains >= numEntriesValidation // 2) or (isMalicious == 1 and numMaliciousDomains >= numEntriesValidation // 2)):
            i += 1
            continue
        
        try:
            shannonEntropy = DomainCalculations.ShannonEntropy(domain)
            vowelConsonantRatio = DomainCalculations.VowelConsonantRatio(domain)
            longestConsecutiveConsonants = DomainCalculations.LongestConsecutiveConsonants(domain)
            dictionaryCount = DomainCalculations.DictionaryCount(domain, dictionaryPath)
            bigramsCount = DomainCalculations.NGramDistribution(domain, bigramsPath)
            trigramsCount = DomainCalculations.NGramDistribution(domain, trigramsPath)
            totalLength = DomainCalculations.TotalLength(domain)
            digitPercentage = DomainCalculations.DigitPercentage(domain)
            longestConsecutiveDigits = DomainCalculations.LongestConsecutiveDigits(domain)
            letterDigitSymbolTransitionCount = DomainCalculations.LetterDigitSymbolTransitionCount(domain)
            indexOfCoincidence = DomainCalculations.IndexOfCoincidence(domain)
            
            cursor.execute("""
                INSERT OR IGNORE INTO domainsValidation (
                    domain,
                    shannonEntropy,
                    vowelConsonantRatio,
                    longestConsecutiveConsonants,
                    dictionaryCount,
                    bigramCount,
                    trigramCount,
                    totalLength,
                    digitPercentage,
                    longestConsecutiveDigits,
                    letterDigitSymbolTransitionCount,
                    indexOfCoincidence,
                    isMalicious
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                domain,
                shannonEntropy,
                vowelConsonantRatio,
                longestConsecutiveConsonants,
                dictionaryCount,
                bigramsCount,
                trigramsCount,
                totalLength,
                digitPercentage,
                longestConsecutiveDigits,
                letterDigitSymbolTransitionCount,
                indexOfCoincidence,
                isMalicious
            ))
            
            if cursor.rowcount > 0:
                if isMalicious == 0:
                    numBenignDomains += 1
                else:
                    numMaliciousDomains += 1
            
        except Exception as e:
            print(f"Warning : {e}")
            #print(f"{domain} is repeated")
        finally:
            i += 1
        
    conn.commit()

GenerateTestingDomains()
GenerateValidationDomains()

notification.notify(
    title="SQL generation complete",
    timeout=10
)