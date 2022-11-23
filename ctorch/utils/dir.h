#ifndef _DIR_H
#define _DIR_H

#include <vector>
#include <unistd.h>
#include <dirent.h>

using namespace std;

void list_dir(const char* parent, vector<string> & dirlist) {
    // list all the directory in this path
    DIR *pDir;
    struct dirent *ptr;

    if (!(pDir = opendir(parent))) {
        return;
    }

    while ((ptr = readdir(pDir)) != NULL) {
        string sub_file = string(parent) + "/" + ptr->d_name;
        // string sub_file = ptr->d_name;
        if (ptr->d_type != 8 && ptr->d_type != 4) {
            continue;
        } else if (ptr->d_type == 4) {
            if (sub_file.c_str()[sub_file.length()-1] != '.') {
                dirlist.push_back(ptr->d_name);
            }
        }
    }
}

void list_file(const char* parent, vector<string> & dirlist, const vector<string> & types) {
    // list all the files in this path
    DIR *pDir;
    struct dirent *ptr;

    if (!(pDir = opendir(parent))) {
        return;
    }

    while ((ptr = readdir(pDir)) != NULL) {
        string sub_file = string(parent) + "/" + ptr->d_name;
        // string sub_file = ptr->d_name;
        if (ptr->d_type != 8 && ptr->d_type != 4) {
            continue;
        } else if (ptr->d_type == 8) {
            bool is_in = false;
            for (int i=0; i<types.size(); i++) {
                if (strstr(ptr->d_name, types[i].c_str())) {
                    is_in = true;
                    break;
                }
            }
            if (is_in) {
                dirlist.push_back(ptr->d_name);
            }
        }
    }
}

#endif
