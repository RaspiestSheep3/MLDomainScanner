// GeneticRouting.cpp : Defines the entry point for the application.
//

#include "Neuroevolution.h"
#include "NeuroevolutionBase.h"

using namespace std;
using namespace std::chrono;

//Settings
int NUM_OF_MODELS = 100;
int NUM_MODELS_FOR_CROSSOVER = 10;
double MUTATION_CHANCE = 0.1;
double MUTATION_SD = 0.1;
int NUM_OF_ITERATIONS = 10000;
int NUM_ENTRIES_FOR_TRAINING = 500;
int GENERATIONS_PER_SHUFFLE = 50;

//Database input
string dbPath;

//Classes and structs
struct DomainEntry {
	float features[6];
	float isMalicious;
};

random_device random;
mt19937 generator(random());
uniform_int_distribution uniformInt(0, NUM_MODELS_FOR_CROSSOVER - 1);
auto rng = default_random_engine{};

//General variables
int main()
{
	cout << "Database Path : ";
	cin >> dbPath;

	vector<unique_ptr<Model>> models;
	vector<tuple<int, double>> outputs = {};

	for (int i = 0; i < NUM_OF_MODELS; i++) {
		auto model = make_unique<Model>();

		model->AddDenseLayer(6, Activation::TANH);
		
		/*model->AddDenseLayer(16, Activation::SIGMOID);
		model->AddDenseLayer(32, Activation::SIGMOID);
		model->AddDenseLayer(32, Activation::SIGMOID);
		model->AddDenseLayer(8, Activation::SIGMOID);*/
		
		model->AddDenseLayer(12, Activation::SIGMOID);

		model->AddDenseLayer(1, Activation::SIGMOID);

		model->GenerateRawList();

		models.push_back(std::move(model));
		outputs.push_back({ i, 0 });
	}


	cout << "Data start" << endl;
	//Producing the training data
	vector<DomainEntry> inputs = {};
	sqlite3* db;
	
	if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
		std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return 1;
	}

	sqlite3_stmt* countStmt;
	const char* countSql = "SELECT COUNT(*) FROM domains;";
	size_t totalRows = 0;

	if (sqlite3_prepare_v2(db, countSql, -1, &countStmt, nullptr) == SQLITE_OK) {
		if (sqlite3_step(countStmt) == SQLITE_ROW) {
			totalRows = static_cast<size_t>(sqlite3_column_int64(countStmt, 0));
		}
		sqlite3_finalize(countStmt);
	}

	if (totalRows > 0) {
		inputs.reserve(totalRows);
	}

	const char* query = "SELECT shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, dictionaryCount, bigramCount, trigramCount, isMalicious FROM domains";
	sqlite3_stmt * stmt;

	if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
		std::cerr << "Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
		return 1;
	}

	while (sqlite3_step(stmt) == SQLITE_ROW) {
		const unsigned char* domain = sqlite3_column_text(stmt, 0);
		
		DomainEntry result;

		result.features[0] = static_cast<float>(sqlite3_column_double(stmt, 0));
		result.features[1] = static_cast<float>(sqlite3_column_double(stmt, 1));
		result.features[2] = static_cast<float>(sqlite3_column_double(stmt, 2));
		result.features[3] = static_cast<float>(sqlite3_column_double(stmt, 3));
		result.features[4] = static_cast<float>(sqlite3_column_double(stmt, 4));
		result.features[5] = static_cast<float>(sqlite3_column_double(stmt, 5));
		result.isMalicious = static_cast<float>(sqlite3_column_int(stmt, 6));

		inputs.push_back(result);
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);

	cout << "Training start" << endl;

	for (int h = 0; h < NUM_OF_ITERATIONS; h++) {

		auto start = high_resolution_clock::now();

		for (int i = 0; i < outputs.size(); i++) {
			get<0>(outputs[i]) = i;
		}

		//Shuffling every n generations
		if(h % GENERATIONS_PER_SHUFFLE == 0) ranges::shuffle(inputs, rng);

		//cout << "Testing phase" << endl;
		#pragma omp parallel for schedule(static)
		for (int j = 0; j < models.size(); j++) {
			vector<double> modelInputs = {};
			vector<double> modelOutputs = {0};
			double offset = 0;

			//Testing phase
			//cout << "Marker 1" << endl;

			for (int i = 0; i < NUM_ENTRIES_FOR_TRAINING; i++) {
				if (modelInputs.size() > 0) modelInputs.clear();
				if (modelOutputs.size() > 1) modelOutputs = { 1 };

				//cout << "Marker 2" << endl;

				const auto& set = inputs[i];

				for (int k = 0; k < 6; k++) modelInputs.push_back(set.features[k]);
				
				//cout << "Marker 2.1" << endl;
				//cout << modelInputs.size() << endl;
				
				models[j]->RunAlgorithm(&modelInputs, &modelOutputs);

				//cout << "Marker 3" << endl;

				offset += fabs(set.isMalicious - modelOutputs[0]);
			}

			get<1>(outputs[j]) = offset;
		}

		auto testingEnd = high_resolution_clock::now();

		//cout << "Evolution phase" << endl;
		//Evolution phase
		sort(outputs.begin(), outputs.end(),
			[](const tuple<int, double>& a, const tuple<int, double>& b) { return get<1>(a) < get<1>(b); }
		);

		//cout << "Most effective score : " << get<1>(outputs[0]) << endl;

		vector<unique_ptr<Model>> newGeneration = {};

		int bestModelIdx = get<0>(outputs[0]);

		//cout << "Best model index : " << bestModelIdx << endl;

		newGeneration.push_back(models[bestModelIdx]->Clone());

		//cout << outputs.size() << " Outputs size " << endl;

		while(newGeneration.size() < NUM_OF_MODELS) {
			int parent1Idx = get<0>(outputs[uniformInt(generator)]);
			int parent2Idx = get<0>(outputs[uniformInt(generator)]);

			while (parent2Idx == parent1Idx && NUM_MODELS_FOR_CROSSOVER > 1) {
				parent2Idx = get<0>(outputs[uniformInt(generator)]);
			}

			auto child = models[parent1Idx]->Clone();

			child->Crossover(models[parent2Idx].get(), MUTATION_CHANCE, MUTATION_SD);
			child->GenerateRawList();

			newGeneration.push_back(std::move(child));
		}

		models = std::move(newGeneration);

		auto evolutionEnd = high_resolution_clock::now();

		if ((h + 1) % 10 == 0) {
			// Cast directly to double seconds/milliseconds to prevent 0ms truncation
			duration<double, std::milli> testingDuration = testingEnd - start;
			duration<double, std::milli> evolutionDuration = evolutionEnd - testingEnd;

			cout << "Iteration : " << h + 1 << "/" << NUM_OF_ITERATIONS
				<< " | Best Loss : " << get<1>(outputs[0]) << "\n"
				<< "  -> Testing: " << testingDuration.count() << " ms\n"
				<< "  -> Evolution: " << evolutionDuration.count() << " ms" << endl;
		}
	}

	sort(outputs.begin(), outputs.end(),
		[](const tuple<int, double>& a, const tuple<int, double>& b) { return get<1>(a) < get<1>(b); }
	);

	cout << "Lowest loss : " << get<1>(outputs[0]) << endl;

	const auto filepath = filesystem::current_path() / "Models" / "1.json";

	cout << "filepath exists?: " << filepath << endl;

	models[get<0>(outputs[0])]->SaveModelWeights(filepath);
	//for (auto output : outputs) cout << get<1>(output) << endl;

	return 0;
}
