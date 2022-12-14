#ifndef _DATASET_H
#define _DATASET_H

#include <string>
#include <opencv2/opencv.hpp>
#include "../tensor/tensor.h"
#include "../utils/dir.h"
#include <random>

using namespace std;
using namespace cv;


template <class T>
class Dataset {
public:

    virtual bool next(Tensor_<T> & data, Tensor_<T> & label) = 0;
    virtual void reset() = 0;
    
    virtual int* _size() = 0;
    virtual int _nsize() = 0;
};

template <class T>
class ImageFolder: public Dataset<T> {
private:
    char datapath[256];
    vector<string> classes; 
    int n_classes;

    vector<string> *filenames;
    int* n_files;

    int* size;
    int nsize;

    int classes_ptr;
    int* file_ptr;

public:
    ImageFolder(int* size, int nsize, const char* datapath, bool is_shuffle = false);
    ~ImageFolder();

    bool next(Tensor_<T> & data, Tensor_<T> & label);
    void reset() {
        this->classes_ptr = 0; 
        delete_s(this->file_ptr); this->file_ptr = new int[n_classes]; // safely deleted
        for (int i = 0; i<n_classes; i++) this->file_ptr[i] = 0;
    }

    int* _size() {return size;}
    int _nsize() {return nsize;}
};



template<class T>
ImageFolder<T>::ImageFolder(int* size, int nsize, const char* datapath, bool is_shuffle) {
    strcpy(this->datapath, datapath);
    this->size = size;
    this->nsize = nsize;
    
    list_dir(datapath, classes);
    sort(classes.begin(), classes.end(), [](string a, string b) {return a.c_str()[a.size()-1] < b.c_str()[b.size()-1];});

    this->classes_ptr = 0;

    n_classes = classes.size();
    n_files = new int[n_classes]; // safely deleted
    this->file_ptr = new int[n_classes]; // safely deleted
    for (int i = 0; i<n_classes; i++) this->file_ptr[i] = 0;

    filenames = new vector<string>[n_classes]; // safely deleted

    vector<string> types;
    types.push_back(string(".png"));

    for (int i=0; i<n_classes; i++) {
        string dirpath = string(datapath) + "/" + classes[i];
        list_file(dirpath.c_str(), filenames[i], types);
        if (is_shuffle) {
            shuffle(filenames[i].begin(), filenames[i].end(), default_random_engine());
        }
        n_files[i] = filenames[i].size();
    }
}

template<class T>
ImageFolder<T>::~ImageFolder() {
    delete_s(filenames);
    delete_s(n_files);
    delete_s(this->file_ptr);
}

template<class T>
bool ImageFolder<T>::next(Tensor_<T> & data, Tensor_<T> & label) {
    /*
    if (this->classes_ptr >= n_classes - 1) {
        this->classes_ptr = 0;
    } else {
        this->classes_ptr++;
    }
    */

   this->classes_ptr = rand() % n_classes;

    for (int n=0; n<n_classes-1; n++) {
        if (this->file_ptr[this->classes_ptr] >= n_files[this->classes_ptr]) {
            this->classes_ptr = (this->classes_ptr + 1) % n_classes;
        } else {
            string file = string(datapath) + "/" + classes[this->classes_ptr] + "/" +filenames[this->classes_ptr][this->file_ptr[this->classes_ptr]];
            data = Tensor_<T>(size, nsize);
            int label_size[] = {1};
            label = Tensor_<T>(label_size, 1);
            label.data[0] = this->classes_ptr;

            Mat img;

            if (size[0] == 1) {
                img = imread(file, IMREAD_GRAYSCALE);
            } else if (size[0] == 3) {
                img = imread(file, IMREAD_COLOR);
            } else {
                img = imread(file, IMREAD_UNCHANGED);
            }

            Mat img_resized;

            resize(img, img_resized, Size(size[1], size[2]), INTER_LINEAR);

            for (int i=0; i<data.nelement(); i++) {
                data.index(i) = img_resized.data[i] / 255.;
            }

            this->file_ptr[this->classes_ptr]++;

            // printf("%s\n", file.c_str());

            return true;
        }
    }
    return false;
}

#endif
