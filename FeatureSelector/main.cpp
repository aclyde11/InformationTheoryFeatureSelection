#include <iostream>

#include <Eigen/Dense>
#include <ctime>
#include <random>
#include <chrono>
#include <vector>
#include <fstream>
#include <cxxopts.hpp>
#include "mutualinformationlib/mi_lib.h"

using Eigen::VectorXi;


cxxopts::Options parser_args() {
    cxxopts::Options options("feature_selector", "Performs MDLP Discretization and JMIM feature selection (");
    options
            .positional_help("[optional args]")
            .show_positional_help();

    options.add_options()
            ("v,verbose", "Enable verbose output")
            ("f,file", "File name", cxxopts::value<std::string>())
            ("y,y_loc", "Y column location in f", cxxopts::value<int>())
            ("d,discretize", "Discretize input (must be discrete if not enabled)")
            ("k,k_features", "Number of features select", cxxopts::value<int>())
            ("o,out", "directory for output files", cxxopts::value<std::string>())
            ("h,headers", "Table contains headers")
            ("help", "Print help");
    cxxopts::value<std::string>()->default_value("value");
    return options;
}

// read CSV of ints.
std::pair<std::vector<VectorXi>, std::vector<std::string>> load_csv_int(const std::string &path, bool headers = false) {
    std::ifstream indata;
    indata.open(path);
    std::string line;

    VectorXi y;
    std::vector<int> values;
    std::vector<std::string> header;
    uint rows = 0;
    uint cols = 0;
    if (headers) {
        while (rows == 0 && std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                header.push_back(cell);
            }
            ++rows;
        }
    }

    rows = 0;
    cols = 0;

    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stoi(cell));
            cols++;
        }
        ++rows;
    }
    cols = cols / rows;
    std::vector<VectorXi> features(cols);
    for (int i = 0; i < cols; i++) {
        features[i].resize(rows);
    }

    int f;
    for (int i = 0; i < values.size(); i++) {
        f = i % cols;
        features[f](i / cols) = values[i];
    }
    return std::make_pair(features, header);
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>>
load_csv_double(const std::string &path, bool headers = false) {
    std::ifstream indata;
    indata.open(path);
    std::string line;


    VectorXi y;
    std::vector<double> values;
    std::vector<std::string> header;
    uint rows = 0;
    uint cols = 0;
    if (headers) {
        while (rows == 0 && std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, ',')) {
                header.push_back(cell);
            }
            ++rows;
        }
    }

    rows = 0;
    cols = 0;

    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
            cols++;
        }
        ++rows;
    }
    cols = cols / rows;
    std::vector<std::vector<double>> features(cols);
    for (int i = 0; i < cols; i++) {
        features[i].resize(rows);
    }

    int f;
    for (int i = 0; i < values.size(); i++) {
        f = i % cols;
        features[f][i / cols] = values[i];
    }
    return std::make_pair(features, header);
}

struct Config {

    Config() {

    }

    Config(const std::string &outfiles, const std::string &inFile, int yLoc, bool verbose, bool headers, int k)
            : outfiles(outfiles), in_file(inFile), y_loc(yLoc), verbose(verbose), headers(headers), k(k) {}

    const std::string &getOutfiles() const {
        return outfiles;
    }

    const std::string &getInFile() const {
        return in_file;
    }

    int getYLoc() const {
        return y_loc;
    }

    bool isVerbose() const {
        return verbose;
    }

    bool isDiscretize() const {
        return discretize;
    }

    bool isHeaders() const {
        return headers;
    }

    int getK() const {
        return k;
    }

    void setOutfiles(const std::string &outfiles) {
        Config::outfiles = outfiles;
    }

    void setInFile(const std::string &inFile) {
        in_file = inFile;
    }

    void setYLoc(int yLoc) {
        y_loc = yLoc;
    }

    void setVerbose(bool verbose) {
        Config::verbose = verbose;
    }

    void setHeaders(bool headers) {
        Config::headers = headers;
    }

    void setK(int k) {
        Config::k = k;
    }

    void setDiscX(const std::vector<VectorXi> &discX) {
        disc_X = discX;
    }

    void setInput(const std::vector<std::vector<double>> &input) {
        Config::input = input;
    }

    void setYVec(const std::vector<int> &yVec) {
        y_vec = yVec;
    }

    void setY(const VectorXi &y) {
        Config::y = y;
    }

    void setDiscretize(bool x) {
        discretize = x;
    }

public:
    std::vector<std::string> header_names;
private:
    //config options
    std::string outfiles;
    std::string in_file;
    int y_loc;
    bool verbose;
    bool headers;
    bool discretize;
    int k;

    //frames:
    std::vector<VectorXi> disc_X;
    std::vector<std::vector<double>> input;
    std::vector<int> y_vec;
    VectorXi y;
};


std::vector<VectorXi> run_discretizer(std::vector<std::vector<double>> X, std::vector<int> y) {


}

template<typename T>
Config parseConfig(T result) {
    Config config{};

    if (result.count("file")) {
        config.setInFile(result["file"].template as<std::string>());
    } else {
        std::cout << "ERROR: Must provide input file" << std::endl;
        exit(1);
    }

    if (result.count("k")) {
        config.setK(result["k"].template as<int>());
    } else {
        std::cout << "ERROR: Must provide number of features to output." << std::endl;
        exit(1);
    }

    if (result.count("out")) {
        config.setOutfiles(result["out"].template as<std::string>());
    } else {
        config.setOutfiles("out/");
    }

    if (result.count("y_loc")) {
        config.setYLoc(result["y_loc"].template as<int>());
    } else {
        std::cout << "ERROR: Must provide location of y column" << std::endl;
        exit(1);
    }


    config.setDiscretize(result["discretize"].template as<bool>());
    config.setVerbose(result["verbose"].template as<bool>());
    config.setHeaders(result["headers"].template as<bool>());
    return config;
}


int main(int argc, char **argv) {
    auto options = parser_args();
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    auto config = parseConfig(result);

    if (config.isDiscretize()) {
        // read double csv
        std::vector<std::vector<double>> X;


        if (config.isVerbose()) {
            std::cout << "Reading file: " << std::endl;
        }
        std::tie(X, config.header_names) = load_csv_double(config.getInFile(), config.isHeaders());
        if (config.isVerbose()) {
            std::cout << "done." << std::endl;
        }

        std::vector<double> y_ = X[config.getYLoc()];
        X.erase(X.begin() + config.getYLoc());
        std::vector<int> y(y_.begin(), y_.end());
        std::vector<VectorXi> X_disc(X.size());
        for (auto &i : X_disc)
            i.resize(y.size());

        Eigen::Map<Eigen::VectorXi> y_eigen(&y[0], y.size());


        if (config.isVerbose()) {
            std::cout << "Starting disc: " << std::endl;
        }
        for (int i = 0; i < X.size(); i++) {
            auto cut_points = mi::MDLPDiscretize(X[i], y);
            X_disc[i] = mi::bin(X[i], cut_points);
        }

        if (config.isVerbose()) {
            std::cout << "Ending disc: " << std::endl;
        }

        // disciretize
        // output dis file
        // do jmim

        if (config.isVerbose()) {
            std::cout << "Starting JMIM: " << std::endl;
        }
        auto p = mi::JMIM(X_disc, y_eigen, config.getK());

        if (config.isVerbose()) {
            std::cout << "Ending JMIM: " << std::endl;
        }
        // output features to list
        std::vector<bool> feature_mask = p.first;
        std::vector<int> ordering = p.second;

        for (int i : ordering) {
            std::cout << i << std::endl;
        }
    } else {
        std::cout << "Only doing disc for now." << std::endl;
    }

    return 0;
}