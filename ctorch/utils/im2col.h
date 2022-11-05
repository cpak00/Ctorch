#ifndef _IMG2COL_H_
#define _IMG2COL_H_

#include "../tensor/tensor.h"
#include "../tensor/matrix.h"

template <typename T>
Tensor_<T> im2col(Tensor_<T> act, int k_size, int stride, int padding) {
    assert(act.ndim() == 4);
    assert(padding == 1 || padding == 0);

    int* size = act.size();

    int batch_ch = size[0];
    int in_ch = size[1];
    int act_h = size[2];
    int act_w = size[3];

    int height_col = (act_h + 2*padding - k_size) / stride + 1;
    int width_col = (act_w + 2*padding - k_size) / stride + 1;

    int col_ch = in_ch * k_size * k_size;

    int size_o[] = {batch_ch, height_col, width_col, col_ch};
    Tensor_<T> output(size_o, 4);

    for (int i = 0; i < batch_ch; i++) {
        for (int c = 0; c < col_ch; c++) {
            int w_offset = c % k_size - padding;
            int h_offset = (c / k_size) % k_size - padding;
            int c_im = c / k_size / k_size;
            for (int h = 0; h < height_col; h++) {
                for (int w = 0; w < width_col; w++) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    int ind_a[] = {i, c_im, im_row, im_col};
                    int ind_o[] = {i, h, w, c};
                    output.index(ind_o) = act.get(ind_a);
                    // output[col_index] = activation.index(ind);
                }
            }
        }   
    }

    return output;
}

template <typename T>
void col2im(Tensor_<T> col, Tensor_<T> & im, int k_size, int stride, int padding) {
    assert(im.ndim() == 4);
    assert(padding == 1 || padding == 0);

    int* size = im.size();

    int batch_ch = size[0];
    int in_ch = size[1];
    int act_h = size[2];
    int act_w = size[3];

    int height_col = (act_h + 2*padding - k_size) / stride + 1;
    int width_col = (act_w + 2*padding - k_size) / stride + 1;

    int col_ch = in_ch * k_size * k_size;

    int size_o[] = {batch_ch, height_col, width_col, col_ch};
    // Tensor_<T> output(size_o, 4);

    for (int i = 0; i < batch_ch; i++) {
        for (int c = 0; c < col_ch; c++) {
            int w_offset = c % k_size - padding;
            int h_offset = (c / k_size) % k_size - padding;
            int c_im = c / k_size / k_size;
            for (int h = 0; h < height_col; h++) {
                for (int w = 0; w < width_col; w++) {
                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;
                    int col_index = (c * height_col + h) * width_col + w;
                    int ind_a[] = {i, c_im, im_row, im_col};
                    int ind_o[] = {i, h, w, c};
                    // output.index(ind_o) = act.get(ind_a);
                    im.index(ind_a) += col.get(ind_o);
                }
            }
        }   
    }
}

#endif
