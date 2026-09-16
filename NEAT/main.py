import neat
import pickle
import sqlite3

#Hyperparameters
domainsPerFitness = 500
falsePositiveMultiplier = 5
falseNegativeMultiplier = 2
connectionPenalty = 0.01

dbPath = input("DB Path : ")
conn = sqlite3.connect(dbPath)
cursor = conn.cursor()

def eval_genomes(genomes, config):
    cursor.execute(f"""
        SELECT * FROM domains
        ORDER BY RANDOM()
        LIMIT {domainsPerFitness}
    """)
            
    rows = cursor.fetchall()
    
    for genome_id, genome in genomes:
        # Create a neural network from this genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = domainsPerFitness

        for row in rows:
            modelInput = tuple(row[1:7])
            res = net.activate(modelInput)[0] #Low => safe, high => malicious
            
            #True  and true negatives
            if((res <= 0.5 and row[-1] == 0) or (res >= 0.5 and row[-1] == 1)):
                fitness -= abs(row[-1] - res)
                
            #False positive (we think an actually safe domain is malicious)
            elif(row[-1] == 0 and res >= 0.5):
                fitness -= abs(row[-1] - res) * falsePositiveMultiplier
            
            #False negative (we think an actually malicious domain is safe)
            elif(row[-1] == 1 and res <= 0.5):
                fitness -= abs(row[-1] - res) * falseNegativeMultiplier
        
        #Connection penalty
        _, numConns = genome.size()
        
        fitness -= numConns * connectionPenalty
        
        genome.fitness = fitness

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'MLConfig')

# Create population
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))

# Run evolution for up to 300 generations
winner = p.run(eval_genomes, 7500)

# Test the winner
print('\nBest genome:\n{!s}'.format(winner))

with open("BestDNSModel.pkl", "wb") as f:
    pickle.dump(winner, f)