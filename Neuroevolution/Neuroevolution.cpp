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
double MUTATION_SD = 0.05;
int NUM_OF_ITERATIONS = 20000;
int NUM_ENTRIES_PER_TRAIN_STEP = 500;
int GENERATIONS_PER_SHUFFLE = 50;
int NUM_ENTRIES_FOR_TRAINING = 90000;
int NUM_ENTRIES_FOR_EVALUATION = 10000;

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

vector<DomainEntry> testingResults;

double Evaluate(Model* model) {
	vector<double> modelInputs = {};
	vector<double> modelOutputs = { 0 };
	double offset = 0;

	//Testing phase
	//cout << "Marker 1" << endl;

	auto start = high_resolution_clock::now();
	for (int i = 0; i < NUM_ENTRIES_FOR_EVALUATION; i++) {
		if (modelInputs.size() > 0) modelInputs.clear();
		if (modelOutputs.size() > 1) modelOutputs = { 1 };

		//cout << "Marker 2" << endl;

		const auto& set = testingResults[i];

		for (int k = 0; k < 6; k++) modelInputs.push_back(set.features[k]);

		//cout << "Marker 2.1" << endl;
		//cout << modelInputs.size() << endl;

		model->RunAlgorithm(&modelInputs, &modelOutputs);

		//cout << "Marker 3" << endl;

		offset += fabs(set.isMalicious - modelOutputs[0]);
	}
	auto end = high_resolution_clock::now();
	duration<double, milli> duration = end - start;

	/*cout << "Testing Results : " << "\n"
		<< " Average Offset : " << offset / NUM_ENTRIES_FOR_EVALUATION << "\n"
		<< " Time Taken     : " << duration << endl;
	*/

	return offset / NUM_ENTRIES_FOR_EVALUATION;
}

void Train()
{
	vector<unique_ptr<Model>> models;
	vector<tuple<int, double>> outputs = {};

	string loadBuffer;
	cout << "Load old models? (Y/N) : ";
	cin >> loadBuffer;
	auto filepath = filesystem::current_path() / "Models" / "1.json";

	for (int i = 0; i < NUM_OF_MODELS; i++) {
		auto model = make_unique<Model>();

		model->AddDenseLayer(6, Activation::TANH);
		
		/*model->AddDenseLayer(16, Activation::SIGMOID);
		model->AddDenseLayer(32, Activation::SIGMOID);
		model->AddDenseLayer(32, Activation::SIGMOID);
		model->AddDenseLayer(8, Activation::SIGMOID);*/
		
		model->AddDenseLayer(12, Activation::SIGMOID);
		model->AddDenseLayer(6, Activation::SIGMOID);

		model->AddDenseLayer(1, Activation::SIGMOID);

		model->GenerateRawList();

		if (loadBuffer == "Y") {
			filepath = filesystem::current_path() / "Models" / "Training" / (to_string(i) + ".json");
			model->LoadModelFromWeights(filepath);
		}

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
	}

	inputs.reserve(NUM_ENTRIES_FOR_TRAINING);

	string queryStr = ("SELECT shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, dictionaryCount, bigramCount, trigramCount, isMalicious FROM domains ORDER BY domain ASC LIMIT " + to_string(NUM_ENTRIES_FOR_TRAINING));
	const char* query = queryStr.c_str();
	sqlite3_stmt * stmt;

	if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
		std::cerr << "Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
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

			for (int i = 0; i < NUM_ENTRIES_PER_TRAIN_STEP; i++) {
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
				<< " | Best Loss : " << Evaluate(models[bestModelIdx].get()) << "\n"
				<< "  -> Testing: " << testingDuration.count() << " ms\n"
				<< "  -> Evolution: " << evolutionDuration.count() << " ms" << endl;
		}
	}

	sort(outputs.begin(), outputs.end(),
		[](const tuple<int, double>& a, const tuple<int, double>& b) { return get<1>(a) < get<1>(b); }
	);

	filepath = filesystem::current_path() / "Models" / "1.json";

	cout << "filepath exists?: " << filepath << endl;

	models[get<0>(outputs[0])]->SaveModelWeights(filepath);

	//We have to save all models for later training
	
	for (int i = 0; i < NUM_OF_MODELS; i++) {
		filepath = filesystem::current_path() / "Models" / "Training" / (to_string(i) + ".json");
		models[get<0>(outputs[i])]->SaveModelWeights(filepath);
	}
}

int main() {
	//Setup for training
	cout << "Database Path : ";
	cin >> dbPath;

	sqlite3* db;

	if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
		std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
	}

	testingResults.reserve(NUM_ENTRIES_FOR_EVALUATION);

	string queryStr = ("SELECT shannonEntropy, vowelConsonantRatio, longestConsecutiveConsonants, dictionaryCount, bigramCount, trigramCount, isMalicious FROM domains ORDER BY domain DESC LIMIT " + to_string(NUM_ENTRIES_FOR_EVALUATION));
	const char* query = queryStr.c_str();
	sqlite3_stmt* stmt;

	if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
		std::cerr << "Failed to prepare query: " << sqlite3_errmsg(db) << std::endl;
		sqlite3_close(db);
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

		testingResults.push_back(result);
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);

	string buffer;
	cout << "Train mode? (Y/N) : ";
	cin >> buffer;

	if (buffer == "Y") Train();
	else {
		auto model = make_unique<Model>();

		model->AddDenseLayer(6, Activation::TANH);

		model->AddDenseLayer(12, Activation::SIGMOID);
		model->AddDenseLayer(6, Activation::SIGMOID);

		model->AddDenseLayer(1, Activation::SIGMOID);

		model->GenerateRawList();

		model->LoadModelFromWeights(filesystem::current_path() / "Models" / "1.json");

		double avgOffset = Evaluate(model.get());

		cout << "Average offset : " << avgOffset << endl;
	}
	return 0;
}