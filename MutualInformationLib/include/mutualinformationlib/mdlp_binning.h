//
// Created by Austin Clyde on 2019-05-22.
//

#ifndef INFORMATIONTHEORYFEATURESELECTION_MDLP_BINNING_H
#define INFORMATIONTHEORYFEATURESELECTION_MDLP_BINNING_H

#include <vector>
#include <set>
#include <limits>
#include "mutual_information_calculations.h"

namespace arr_utils {
    template<class T>
    struct index_cmp {
        index_cmp(const T arr) : arr(arr) {}

        bool operator()(const size_t a, const size_t b) const {
            return arr[a] < arr[b];
        }

        const T arr;
    };

    // This implementation is O(n), but also uses O(n) extra memory
    template<class T>
    void reorder(
            std::vector<T> &unordered,
            std::vector<size_t> const &index_map,
            std::vector<T> &ordered) {
        // copy for the reorder according to index_map, because unsorted may also be
        // sorted
        std::vector<T> copy = unordered;
        ordered.resize(index_map.size());
        for (int i = 0; i < index_map.size(); i++) {
            ordered[i] = copy[index_map[i]];
        }
    }

    template<class T>
    void sort(
            std::vector<T> &unsorted,
            std::vector<T> &sorted,
            std::vector<size_t> &index_map) {
        // Original unsorted index map
        index_map.resize(unsorted.size());
        for (size_t i = 0; i < unsorted.size(); i++) {
            index_map[i] = i;
        }
        // Sort the index map, using unsorted for comparison
        sort(
                index_map.begin(),
                index_map.end(),
                index_cmp<std::vector<T> &>(unsorted));

        sorted.resize(unsorted.size());
        reorder(unsorted, index_map, sorted);
    }

}

namespace mi {



    struct Level {
        int level[3];

        Level() {
            level[0] = 0;
            level[1] = 0;
            level[2] = 0;
        }

        Level(int start, int end, int depth) {
            setLevel(start, end, depth);
        }

        void setLevel(int start, int end, int depth) {
            level[0] = start;
            level[1] = end;
            level[2] = depth;
        }

        inline int start() const {
            return level[0];
        }

        inline int end() const {
            return level[1];
        }

        inline int depth() const {
            return level[2];
        }

    };

    int find_cut(std::vector<int> y, int start, int end) {
        auto length = (double) (end - start);
        double prev_entropy = 100000.0;
        int k = -1;
        double first_half, second_half, curr_entropy;

        for (int ind = start + 1; ind < end; ind++) {
            if (y[ind - 1] == y[ind])
                continue;

            first_half = ((double) (ind - start)) / length * compute_entropy(y, start, ind);
            second_half = ((double) (end - ind)) / length * compute_entropy(y, ind, end);
            curr_entropy = first_half + second_half;
            if (prev_entropy > curr_entropy) {
                prev_entropy = curr_entropy;
                k = ind;
            }
        }
        return k;
    }

    bool reject_split(std::vector<int> const &y, int start, int end, int k) {
        auto N = (double) (end - start);
        int k1, k2, k0;
        double entropy1, entropy2, entropy0;

        std::tie(entropy0, k0) = slice_entropy(y, start, end);
        std::tie(entropy1, k1) = slice_entropy(y, start, k);
        std::tie(entropy2, k2) = slice_entropy(y, k, end);

        double part1 = 1 / N * ((k - start) * entropy1 + (end - k) * entropy2);
        double gain = entropy0 - part1;
        double entropy_diff = k0 * entropy0 - k1 * entropy1 - k2 * entropy2;
        double delta = log(pow(3, k0) - 2) - entropy_diff;
        return gain <= 1 / N * (log(N - 1) + delta);
    }

    double get_cut(std::vector<double> col, int ind) {
        return (col[ind - 1] + col[ind]) / 2.0;
    }


    std::vector<double>
    MDLPDiscretize(std::vector<double> col_, std::vector<int> y_, int min_depth = 0) {
        std::vector<size_t> order;
        std::vector<double> col;
        std::vector<int> y;

        arr_utils::sort(col_, col, order);
        arr_utils::reorder(y_, order, y);

        int num_samples = col.size();
        Level init_level(0, num_samples, 0);
        std::vector<Level> search_intervals;
        std::set<double> cut_points;
        search_intervals.push_back(init_level);
        Level curr_level{};

        while (!search_intervals.empty()) {
            curr_level = search_intervals.back();
            search_intervals.pop_back();

            int k = find_cut(y, curr_level.start(), curr_level.end());

            if ((k == -1) ||
                (curr_level.depth() >= min_depth && reject_split(y, curr_level.start(), curr_level.end(), k))) {
                double front = (curr_level.start() == 0) ? -1.0 * std::numeric_limits<double>::infinity() : get_cut(col,
                                                                                                                    curr_level.start());
                double back = (curr_level.end() == num_samples) ? std::numeric_limits<double>::infinity() : get_cut(col,
                                                                                                                    curr_level.end());
                if (front == back)
                    continue;
                if (front != -1.0 * std::numeric_limits<double>::infinity())
                    cut_points.insert(front);
                if (back != std::numeric_limits<double>::infinity())
                    cut_points.insert(back);
                continue;
            }

            Level left_level(curr_level.start(), k, curr_level.depth() + 1);
            Level right_level(k, curr_level.end(), curr_level.depth() + 1);
            search_intervals.push_back(left_level);
            search_intervals.push_back(right_level);
        }

        std::vector<double> output;
        for (auto pos = cut_points.begin(); pos != cut_points.end(); ++pos) {
            output.push_back(*pos);
        }
        return output;
    }

    VectorXi bin(std::vector<double> const &col, std::vector<double> const &cut_points) {
        VectorXi output(col.size());
        int bins = cut_points.size();
        int tmp;
        for (int i = 0; i < col.size(); i++) {
            for (tmp = 0; tmp < bins; tmp++) {
                if (col[i] <= cut_points[tmp])
                    break;
            }
            output[i] = tmp;
        }
        return output;
    }
}

#endif //INFORMATIONTHEORYFEATURESELECTION_MDLP_BINNING_H
