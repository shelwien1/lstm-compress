/*
 * Convert compressed_sizes.txt and pair_compressed_sizes.txt to TSPLIB format.
 * Outputs TSPLIB EXPLICIT LOWER_DIAG_ROW format for use with TSP solvers.
 *
 * Usage: sizes2tsp [output_file]
 *   If output_file is not specified, writes to stdout.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

struct ProblemData {
    vector<string> items;                    // Item names in sorted order
    map<string, int> individual_sizes;       // Individual sizes
    map<string, int> pair_sizes;             // Pair sizes (symmetric)
};

// Parse individual compressed sizes from file
void parse_individual_sizes(const char* filename, map<string, int>& sizes) {
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

    fprintf(stderr, "Loaded %zu individual items\n", sizes.size());
}

// Parse pair compressed sizes from file
void parse_pair_sizes(const char* filename, map<string, int>& pairs) {
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

        // Store both directions (symmetric)
        string key1 = item_x + "_" + item_y;
        string key2 = item_y + "_" + item_x;
        pairs[key1] = size;
        pairs[key2] = size;
    }

    fprintf(stderr, "Loaded %zu pair entries\n", pairs.size());
}

// Get pair size or compute default if not found
int get_pair_size(const ProblemData& data, const string& item1, const string& item2) {
    if (item1 == item2) {
        return 0;  // Diagonal: distance to self is 0
    }

    string key = item1 + "_" + item2;
    auto it = data.pair_sizes.find(key);

    if (it != data.pair_sizes.end()) {
        return it->second;
    } else {
        // If pair not found, use sum of individual sizes as default
        int size1 = data.individual_sizes.at(item1);
        int size2 = data.individual_sizes.at(item2);
        return size1 + size2;
    }
}

// Output TSPLIB format
void write_tsplib(FILE* out, const ProblemData& data) {
    const size_t n = data.items.size();

    // Header
    fprintf(out, "NAME: CompressedSizes\n");
    fprintf(out, "TYPE: TSP\n");
    fprintf(out, "COMMENT: Pairwise compression sizes from compressed_sizes.txt and pair_compressed_sizes.txt\n");
    fprintf(out, "DIMENSION: %zu\n", n);
    fprintf(out, "EDGE_WEIGHT_TYPE: EXPLICIT\n");
    fprintf(out, "EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW\n");
    fprintf(out, "EDGE_WEIGHT_SECTION\n");

    // Write lower triangular matrix with diagonal
    // Row i has i+1 entries: elements [i][0] through [i][i]
    for (size_t i = 0; i < n; i++) {
        const string& item_i = data.items[i];

        for (size_t j = 0; j <= i; j++) {
            const string& item_j = data.items[j];
            int cost = get_pair_size(data, item_i, item_j);

            fprintf(out, "%d", cost);

            // Add space or newline
            if (j < i) {
                fprintf(out, " ");
            } else {
                fprintf(out, "\n");
            }
        }
    }

    fprintf(out, "EOF\n");
}

int main(int argc, char* argv[]) {
    // Parse command line
    const char* output_filename = nullptr;
    FILE* output = stdout;

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printf("Usage: %s [output_file]\n", argv[0]);
            printf("\nConverts compressed_sizes.txt and pair_compressed_sizes.txt to TSPLIB format.\n");
            printf("\nOptions:\n");
            printf("  output_file          Output file (default: stdout)\n");
            printf("  -h, --help          Show this help message\n");
            printf("\nExample:\n");
            printf("  %s problem.tsp      # Write to problem.tsp\n", argv[0]);
            printf("  %s > problem.tsp    # Write to stdout and redirect\n", argv[0]);
            return 0;
        }

        output_filename = argv[1];
        output = fopen(output_filename, "w");
        if (!output) {
            fprintf(stderr, "Error: Cannot open output file %s\n", output_filename);
            return 1;
        }
    }

    // Load data
    fprintf(stderr, "Loading compressed_sizes.txt...\n");
    ProblemData data;
    parse_individual_sizes("compressed_sizes.txt", data.individual_sizes);

    fprintf(stderr, "Loading pair_compressed_sizes.txt...\n");
    parse_pair_sizes("pair_compressed_sizes.txt", data.pair_sizes);

    // Build sorted item list
    for (const auto& [item, size] : data.individual_sizes) {
        data.items.push_back(item);
    }
    sort(data.items.begin(), data.items.end());

    fprintf(stderr, "\nGenerating TSPLIB format with %zu nodes...\n", data.items.size());
    write_tsplib(output, data);

    if (output_filename) {
        fclose(output);
        fprintf(stderr, "TSPLIB file written to: %s\n", output_filename);
    }

    return 0;
}
