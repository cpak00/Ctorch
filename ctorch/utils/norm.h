
template <typename T>
T mean(T* data, int narray) {
    T res = 0;
    for (int i=0; i<narray; i++) {
        res += data[i];
    }

    return res / (double) narray;
}

template <typename T>
T var(T* data, int narray) {
    T res = 0;
    for (int i=0; i<narray; i++) {
        res += data[i] * data[i];
    }

    return sqrt(res / (double) narray);
}

template <typename T>
void normalize(T* data, int narray, T new_m, T new_var) {
    T m = mean(data, narray);
    T v = var(data, narray);

    for (int i=0; i<narray; i++) {
        data[i] = (data[i] - m + new_m) / v * new_var;
    }
}