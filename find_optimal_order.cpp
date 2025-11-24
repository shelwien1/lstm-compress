/*
 * Find the optimal ordering of items to maximize pairwise compression gains.
 * Uses multiple algorithms: greedy construction, simulated annealing, and genetic algorithm.
 *
 * Optimized C++ version with focus on speed.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// Use fast random number generator
static mt19937 rng(time(nullptr));

// Item index type for fast lookups
using ItemIdx = uint32_t;

// Gain matrix using flat 2D array for cache efficiency
struct GainMatrix {
    vector<float> data;
    size_t size;

    GainMatrix(size_t n) : size(n) {
        data.resize(n * n);
    }

    inline float& operator()(ItemIdx i, ItemIdx j) {
        return data[i * size + j];
    }

    inline float operator()(ItemIdx i, ItemIdx j) const {
        return data[i * size + j];
    }
};

struct ProblemData {
    vector<string> items;                           // Item names
    unordered_map<string, ItemIdx> item_to_idx;     // Name to index
    unordered_map<string, int> individual_sizes;    // Individual sizes
    unordered_map<string, int> pair_sizes;          // Pair sizes (both directions)
    GainMatrix gain_matrix;                         // Precomputed gains

    ProblemData(size_t n) : gain_matrix(n) {}
};

// Parse individual compressed sizes from file
void parse_individual_sizes(const char* filename, unordered_map<string, int>& sizes) {
    ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        size_t dash_pos = line.find(" - ");
        if (dash_pos == string::npos) continue;

        string item_id = line.substr(0, dash_pos);
        int size = atoi(line.c_str() + dash_pos + 3);
        sizes[item_id] = size;
    }
}

// Parse pair compressed sizes from file
void parse_pair_sizes(const char* filename, unordered_map<string, int>& pairs) {
    ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;

        size_t dash_pos = line.find(" - ");
        if (dash_pos == string::npos) continue;

        string items = line.substr(0, dash_pos);
        int size = atoi(line.c_str() + dash_pos + 3);

        size_t underscore_pos = items.find('_');
        if (underscore_pos == string::npos) continue;

        string item_x = items.substr(0, underscore_pos);
        string item_y = items.substr(underscore_pos + 1);

        // Store both directions
        pairs[item_x + "_" + item_y] = size;
        pairs[item_y + "_" + item_x] = size;
    }
}

// Precompute gain matrix for all pairs
void compute_gain_matrix(ProblemData& data) {
    const size_t n = data.items.size();

    for (size_t i = 0; i < n; i++) {
        const string& item1 = data.items[i];
        int size1 = data.individual_sizes[item1];

        for (size_t j = 0; j < n; j++) {
            if (i == j) {
                data.gain_matrix(i, j) = 0;
                continue;
            }

            const string& item2 = data.items[j];
            int size2 = data.individual_sizes[item2];

            string key = item1 + "_" + item2;
            auto it = data.pair_sizes.find(key);
            int pair_size = (it != data.pair_sizes.end()) ? it->second : (size1 + size2);

            float gain = pair_size - (size1 + size2);
            data.gain_matrix(i, j) = gain;
        }
    }
}

// Calculate total gain for a given ordering (hot path - optimized)
inline float evaluate_order(const vector<ItemIdx>& order, const GainMatrix& gain_matrix) {
    float total_gain = 0;
    const size_t n = order.size();

    // Manual loop unrolling for better performance
    size_t i = 0;
    for (; i + 4 <= n - 1; i += 4) {
        total_gain += gain_matrix(order[i], order[i+1]);
        total_gain += gain_matrix(order[i+1], order[i+2]);
        total_gain += gain_matrix(order[i+2], order[i+3]);
        total_gain += gain_matrix(order[i+3], order[i+4]);
    }

    // Handle remaining elements
    for (; i < n - 1; i++) {
        total_gain += gain_matrix(order[i], order[i+1]);
    }

    return total_gain;
}

// Greedy construction from multiple starting points
vector<ItemIdx> greedy_construction(const ProblemData& data, int num_starts = 20) {
    const size_t n = data.items.size();
    vector<ItemIdx> best_order;
    float best_gain = FLT_MAX;

    num_starts = min(num_starts, (int)n);

    for (int start_idx = 0; start_idx < num_starts; start_idx++) {
        vector<ItemIdx> order;
        order.reserve(n);
        order.push_back(start_idx);

        vector<bool> used(n, false);
        used[start_idx] = true;

        while (order.size() < n) {
            ItemIdx last_item = order.back();
            ItemIdx best_item = 0;
            float best_gain_local = FLT_MAX;

            for (ItemIdx item = 0; item < n; item++) {
                if (used[item]) continue;

                float gain = data.gain_matrix(last_item, item);
                if (gain < best_gain_local) {
                    best_gain_local = gain;
                    best_item = item;
                }
            }

            order.push_back(best_item);
            used[best_item] = true;
        }

        float total_gain = evaluate_order(order, data.gain_matrix);
        if (total_gain < best_gain) {
            best_gain = total_gain;
            best_order = std::move(order);
        }
    }

    return best_order;
}

// 2-opt local optimization (hot path - heavily optimized)
vector<ItemIdx> two_opt(vector<ItemIdx> order, const GainMatrix& gain_matrix, int max_iterations = 1000) {
    const size_t n = order.size();
    float current_gain = evaluate_order(order, gain_matrix);
    bool improved = true;
    int iterations = 0;

    while (improved && iterations < max_iterations) {
        improved = false;
        iterations++;

        for (size_t i = 0; i < n - 1; i++) {
            for (size_t j = i + 2; j < n; j++) {
                // Calculate delta without creating new array
                // Remove edges: (i, i+1) and (j, j+1)
                // Add edges: (i, j) and (i+1, j+1)
                float delta = 0;

                if (j < n - 1) {
                    delta -= gain_matrix(order[i], order[i+1]);
                    delta -= gain_matrix(order[j], order[j+1]);
                    delta += gain_matrix(order[i], order[j]);
                    delta += gain_matrix(order[i+1], order[j+1]);
                } else {
                    delta -= gain_matrix(order[i], order[i+1]);
                    delta += gain_matrix(order[i], order[j]);
                }

                if (delta < -1e-6) {  // Improvement found
                    // Reverse segment [i+1, j]
                    reverse(order.begin() + i + 1, order.begin() + j + 1);
                    current_gain += delta;
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }

    return order;
}

// Simulated annealing for optimization
vector<ItemIdx> simulated_annealing(const ProblemData& data,
                                    float initial_temp = 1000.0f,
                                    float cooling_rate = 0.995f,
                                    int max_iterations = 5000) {
    const size_t n = data.items.size();

    // Create random initial order
    vector<ItemIdx> current_order(n);
    for (size_t i = 0; i < n; i++) current_order[i] = i;
    shuffle(current_order.begin(), current_order.end(), rng);

    float current_gain = evaluate_order(current_order, data.gain_matrix);

    vector<ItemIdx> best_order = current_order;
    float best_gain = current_gain;

    float temp = initial_temp;
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    uniform_int_distribution<size_t> idx_dist(0, n - 1);

    for (int iteration = 0; iteration < max_iterations; iteration++) {
        // Generate neighbor by swapping two random elements
        size_t i = idx_dist(rng);
        size_t j = idx_dist(rng);

        if (i == j) continue;

        // Swap
        swap(current_order[i], current_order[j]);
        float new_gain = evaluate_order(current_order, data.gain_matrix);

        float gain_change = new_gain - current_gain;

        // Accept or reject
        if (gain_change < 0 || exp(-gain_change / temp) > dist(rng)) {
            current_gain = new_gain;

            if (current_gain < best_gain) {
                best_order = current_order;
                best_gain = current_gain;
            }
        } else {
            // Revert swap
            swap(current_order[i], current_order[j]);
        }

        temp *= cooling_rate;

        if (temp < 1e-6f) break;
    }

    return best_order;
}

// Genetic algorithm for optimization
vector<ItemIdx> genetic_algorithm(const ProblemData& data,
                                  int population_size = 50,
                                  int generations = 100,
                                  float mutation_rate = 0.1f) {
    const size_t n = data.items.size();

    // Initialize population
    vector<vector<ItemIdx>> population(population_size);
    for (int i = 0; i < population_size; i++) {
        population[i].resize(n);
        for (size_t j = 0; j < n; j++) population[i][j] = j;
        shuffle(population[i].begin(), population[i].end(), rng);
    }

    // Initialize with first individual to ensure we always return something valid
    vector<ItemIdx> best_individual = population[0];
    float best_gain = evaluate_order(best_individual, data.gain_matrix);

    uniform_real_distribution<float> dist(0.0f, 1.0f);
    uniform_int_distribution<size_t> idx_dist(0, n - 1);
    uniform_int_distribution<int> pop_dist(0, population_size - 1);

    for (int generation = 0; generation < generations; generation++) {
        // Evaluate fitness
        vector<float> gains(population_size);
        for (int i = 0; i < population_size; i++) {
            gains[i] = evaluate_order(population[i], data.gain_matrix);

            if (gains[i] < best_gain) {
                best_gain = gains[i];
                best_individual = population[i];
            }
        }

        // Selection (tournament) and create new population
        vector<vector<ItemIdx>> new_population;
        new_population.reserve(population_size);

        for (int i = 0; i < population_size; i++) {
            // Tournament selection
            int idx1 = pop_dist(rng);
            int idx2 = pop_dist(rng);
            int idx3 = pop_dist(rng);

            int winner_idx = idx1;
            if (gains[idx2] < gains[winner_idx]) winner_idx = idx2;
            if (gains[idx3] < gains[winner_idx]) winner_idx = idx3;

            new_population.push_back(population[winner_idx]);
        }

        // Crossover and mutation
        population.clear();
        for (int i = 0; i + 1 < population_size; i += 2) {
            // Ordered crossover
            auto crossover = [&](const vector<ItemIdx>& parent1, const vector<ItemIdx>& parent2) {
                vector<ItemIdx> child(n);

                size_t start = idx_dist(rng);
                size_t end = idx_dist(rng);
                if (start > end) swap(start, end);

                // Copy segment from parent1
                vector<bool> used(n, false);
                for (size_t j = start; j <= end; j++) {
                    child[j] = parent1[j];
                    used[parent1[j]] = true;
                }

                // Fill remaining from parent2 in order
                size_t child_pos = 0;
                for (size_t ptr = 0; ptr < n; ptr++) {
                    if (!used[parent2[ptr]]) {
                        // Find next unfilled position in child
                        while (child_pos < n && (child_pos >= start && child_pos <= end)) {
                            child_pos++;
                        }
                        if (child_pos < n) {
                            child[child_pos] = parent2[ptr];
                            child_pos++;
                        }
                    }
                }

                return child;
            };

            // Mutation
            auto mutate = [&](vector<ItemIdx>& individual) {
                if (dist(rng) < mutation_rate) {
                    size_t i = idx_dist(rng);
                    size_t j = idx_dist(rng);
                    swap(individual[i], individual[j]);
                }
            };

            vector<ItemIdx> child1 = crossover(new_population[i], new_population[i+1]);
            vector<ItemIdx> child2 = crossover(new_population[i+1], new_population[i]);
            mutate(child1);
            mutate(child2);

            population.push_back(std::move(child1));
            population.push_back(std::move(child2));
        }

        // Keep population size constant
        if (population.size() > (size_t)population_size) {
            population.resize(population_size);
        }
    }

    return best_individual;
}

// Run algorithm with 2-opt optimization
template<typename AlgoFunc>
pair<vector<ItemIdx>, float> run_algorithm_with_2opt(
    const char* name,
    AlgoFunc algorithm,
    const ProblemData& data) {

    printf("Running %s...\n", name);

    vector<ItemIdx> order = algorithm();
    float gain_before = evaluate_order(order, data.gain_matrix);
    order = two_opt(std::move(order), data.gain_matrix);
    float gain_after = evaluate_order(order, data.gain_matrix);
    float improvement = gain_before - gain_after;

    printf("  %s: %.0f bytes (2-opt improved by %.0f bytes)\n",
           name, gain_after, improvement);

    return {std::move(order), gain_after};
}

// Hybrid optimization approach
pair<vector<ItemIdx>, float> hybrid_optimization(const ProblemData& data, int time_limit = 60) {
    time_t start_time = time(nullptr);
    const size_t n = data.items.size();

    // Initialize with trivial order to ensure we always have a valid solution
    vector<ItemIdx> best_order(n);
    for (size_t i = 0; i < n; i++) best_order[i] = i;
    float best_gain = evaluate_order(best_order, data.gain_matrix);

    // Greedy construction
    if (time(nullptr) - start_time < time_limit) {
        auto [order, gain] = run_algorithm_with_2opt("greedy construction",
            [&]() { return greedy_construction(data, 20); }, data);
        if (gain < best_gain) {
            best_order = std::move(order);
            best_gain = gain;
        }
    }

    // Simulated annealing
    if (time(nullptr) - start_time < time_limit) {
        auto [order, gain] = run_algorithm_with_2opt("simulated annealing",
            [&]() { return simulated_annealing(data); }, data);
        if (gain < best_gain) {
            best_order = std::move(order);
            best_gain = gain;
        }
    }

    // Genetic algorithm
    if (time(nullptr) - start_time < time_limit) {
        auto [order, gain] = run_algorithm_with_2opt("genetic algorithm",
            [&]() { return genetic_algorithm(data); }, data);
        if (gain < best_gain) {
            best_order = std::move(order);
            best_gain = gain;
        }
    }

    // Multi-start random search with remaining time
    printf("Running multi-start random search...\n");
    int iterations = 0;

    while (time(nullptr) - start_time < time_limit && iterations < 100) {
        vector<ItemIdx> random_order(n);
        for (size_t i = 0; i < n; i++) random_order[i] = i;
        shuffle(random_order.begin(), random_order.end(), rng);

        vector<ItemIdx> order = two_opt(std::move(random_order), data.gain_matrix);
        float gain = evaluate_order(order, data.gain_matrix);

        if (gain < best_gain) {
            best_order = std::move(order);
            best_gain = gain;
            printf("  Random start %d: Improved to %.0f\n", iterations, gain);
        }

        iterations++;
    }

    return {std::move(best_order), best_gain};
}

// Analyze and display results
void analyze_results(const vector<ItemIdx>& best_order, const ProblemData& data) {
    float best_gain = evaluate_order(best_order, data.gain_matrix);

    printf("\n");
    printf("================================================================================\n");
    printf("BEST ORDERING FOUND:\n");
    printf("================================================================================\n");
    printf("Total compression gain: %.0f bytes\n", best_gain);
    printf("Average gain per adjacent pair: %.2f bytes\n", best_gain / (best_order.size() - 1));

    // Calculate total independent size and solid size
    int total_independent = 0;
    for (const auto& item : data.items) {
        total_independent += data.individual_sizes.at(item);
    }

    float total_solid = total_independent + best_gain;
    float compression_ratio = total_solid / total_independent;

    printf("Total independent size: %d bytes\n", total_independent);
    printf("Total solid size: %.0f bytes\n", total_solid);
    printf("Compression ratio: %.4f\n", compression_ratio);

    printf("\nOrdering (%zu items):\n", best_order.size());
    for (size_t i = 0; i < best_order.size(); i++) {
        const string& item = data.items[best_order[i]];
        printf("%3zu: %s (%d bytes)\n", i, item.c_str(), data.individual_sizes.at(item));
    }

    // Show top pairs by gain
    printf("\n");
    printf("================================================================================\n");
    printf("TOP 10 ADJACENT PAIRS BY GAIN:\n");
    printf("================================================================================\n");
    printf("%-20s %10s %10s %10s %10s %10s\n",
           "Pair", "Size X", "Size Y", "Solid", "Gain", "Savings%");
    printf("--------------------------------------------------------------------------------\n");

    struct PairGain {
        string item1, item2;
        int size1, size2, pair_size;
        float gain, savings_pct;
    };

    vector<PairGain> pair_gains;
    for (size_t i = 0; i < best_order.size() - 1; i++) {
        const string& item1 = data.items[best_order[i]];
        const string& item2 = data.items[best_order[i+1]];
        int size1 = data.individual_sizes.at(item1);
        int size2 = data.individual_sizes.at(item2);

        string key = item1 + "_" + item2;
        int pair_size = data.pair_sizes.at(key);
        float gain = pair_size - (size1 + size2);
        float savings_pct = -100.0f * gain / (size1 + size2);

        pair_gains.push_back({item1, item2, size1, size2, pair_size, gain, savings_pct});
    }

    // Sort by gain (most negative first)
    sort(pair_gains.begin(), pair_gains.end(),
         [](const PairGain& a, const PairGain& b) { return a.gain < b.gain; });

    for (size_t i = 0; i < min(pair_gains.size(), size_t(10)); i++) {
        const auto& pg = pair_gains[i];
        string pair_name = pg.item1 + "_" + pg.item2;
        printf("%-20s %10d %10d %10d %10.0f %9.2f%%\n",
               pair_name.c_str(), pg.size1, pg.size2, pg.pair_size, pg.gain, pg.savings_pct);
    }
}

// Save ordering to file
void save_ordering(const char* filename, const vector<ItemIdx>& order, const ProblemData& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open output file %s\n", filename);
        return;
    }

    float best_gain = evaluate_order(order, data.gain_matrix);

    int total_independent = 0;
    for (const auto& item : data.items) {
        total_independent += data.individual_sizes.at(item);
    }
    float total_solid = total_independent + best_gain;
    float compression_ratio = total_solid / total_independent;

    file << "# Optimal ordering of items to maximize pairwise compression gains\n";
    file << "# Total gain: " << best_gain << " bytes\n";
    file << "# Average gain per pair: " << (best_gain / (order.size() - 1)) << " bytes\n";
    file << "# Total independent size: " << total_independent << " bytes\n";
    file << "# Total solid size: " << total_solid << " bytes\n";
    file << "# Compression ratio: " << compression_ratio << "\n\n";

    for (ItemIdx idx : order) {
        file << data.items[idx] << "\n";
    }

    printf("\nOrdering saved to: %s\n", filename);
}

int main() {
    printf("Loading data...\n");

    ProblemData data(0);
    parse_individual_sizes("compressed_sizes.txt", data.individual_sizes);
    parse_pair_sizes("pair_compressed_sizes.txt", data.pair_sizes);

    // Build item list and index mapping
    for (const auto& [item, size] : data.individual_sizes) {
        ItemIdx idx = data.items.size();
        data.items.push_back(item);
        data.item_to_idx[item] = idx;
    }

    // Sort items for consistent ordering
    sort(data.items.begin(), data.items.end());

    // Rebuild index mapping after sort
    data.item_to_idx.clear();
    for (size_t i = 0; i < data.items.size(); i++) {
        data.item_to_idx[data.items[i]] = i;
    }

    printf("Found %zu items\n", data.items.size());
    printf("Found %zu unique pairs\n", data.pair_sizes.size() / 2);

    // Resize and compute gain matrix
    data.gain_matrix = GainMatrix(data.items.size());
    printf("\nPrecomputing gain matrix...\n");
    compute_gain_matrix(data);

    printf("\nFinding optimal ordering using hybrid approach...\n");
    time_t start_time = time(nullptr);
    auto [best_order, best_gain] = hybrid_optimization(data, 60);
    time_t end_time = time(nullptr);

    printf("\nOptimization completed in %ld seconds\n", end_time - start_time);

    // Analyze results
    analyze_results(best_order, data);

    // Save ordering to file
    save_ordering("optimal_order-ds.txt", best_order, data);

    return 0;
}
