#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace sorting {

void bubble_sort(std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (a[j] > a[j + 1]) {
                std::swap(a[j], a[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

void selection_sort(std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n - 1; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (a[j] < a[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            std::swap(a[i], a[min_idx]);
        }
    }
}

void insertion_sort(std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 1; i < n; ++i) {
        const int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

int partition_lomuto(std::vector<int>& a, int l, int r) {
    const int pivot = a[r];
    int i = l - 1;
    for (int j = l; j < r; ++j) {
        if (a[j] <= pivot) {
            ++i;
            std::swap(a[i], a[j]);
        }
    }
    std::swap(a[i + 1], a[r]);
    return i + 1;
}

void quick_sort_impl(std::vector<int>& a, int l, int r) {
    if (l >= r) {
        return;
    }
    const int p = partition_lomuto(a, l, r);
    quick_sort_impl(a, l, p - 1);
    quick_sort_impl(a, p + 1, r);
}

void quick_sort(std::vector<int>& a) {
    if (a.empty()) {
        return;
    }
    quick_sort_impl(a, 0, static_cast<int>(a.size()) - 1);
}

void merge_range(std::vector<int>& a, int l, int m, int r, std::vector<int>& tmp) {
    int i = l;
    int j = m + 1;
    int k = l;
    while (i <= m && j <= r) {
        if (a[i] <= a[j]) {
            tmp[k++] = a[i++];
        } else {
            tmp[k++] = a[j++];
        }
    }
    while (i <= m) {
        tmp[k++] = a[i++];
    }
    while (j <= r) {
        tmp[k++] = a[j++];
    }
    for (int p = l; p <= r; ++p) {
        a[p] = tmp[p];
    }
}

void merge_sort_impl(std::vector<int>& a, int l, int r, std::vector<int>& tmp) {
    if (l >= r) {
        return;
    }
    const int m = l + (r - l) / 2;
    merge_sort_impl(a, l, m, tmp);
    merge_sort_impl(a, m + 1, r, tmp);
    merge_range(a, l, m, r, tmp);
}

void merge_sort(std::vector<int>& a) {
    if (a.empty()) {
        return;
    }
    std::vector<int> tmp(a.size());
    merge_sort_impl(a, 0, static_cast<int>(a.size()) - 1, tmp);
}

void heapify(std::vector<int>& a, int n, int i) {
    int largest = i;
    const int left = 2 * i + 1;
    const int right = 2 * i + 2;
    if (left < n && a[left] > a[largest]) {
        largest = left;
    }
    if (right < n && a[right] > a[largest]) {
        largest = right;
    }
    if (largest != i) {
        std::swap(a[i], a[largest]);
        heapify(a, n, largest);
    }
}

void heap_sort(std::vector<int>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = n / 2 - 1; i >= 0; --i) {
        heapify(a, n, i);
    }
    for (int i = n - 1; i > 0; --i) {
        std::swap(a[0], a[i]);
        heapify(a, i, 0);
    }
}

}  // namespace sorting

using SortFn = void (*)(std::vector<int>&);

bool is_sorted_asc(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i - 1] > a[i]) {
            return false;
        }
    }
    return true;
}

void run_one(const std::string& name, SortFn fn, const std::vector<int>& src) {
    std::vector<int> v = src;
    fn(v);
    assert(is_sorted_asc(v));
    std::cout << name << " ok\n";
}

int main() {
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    std::vector<int> src(1000);
    for (int& x : src) {
        x = dist(rng);
    }

    run_one("bubble_sort", sorting::bubble_sort, src);
    run_one("selection_sort", sorting::selection_sort, src);
    run_one("insertion_sort", sorting::insertion_sort, src);
    run_one("quick_sort", sorting::quick_sort, src);
    run_one("merge_sort", sorting::merge_sort, src);
    run_one("heap_sort", sorting::heap_sort, src);

    std::cout << "all sorting algorithms passed\n";
    return 0;
}
