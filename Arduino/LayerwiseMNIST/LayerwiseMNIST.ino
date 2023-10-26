#include "arducampico.h"
#include <stdio.h>
#include "stdlib.h"

int iteration = 0;
int signal_pin = 13;   // output a high signal to indicate the execution is running

int16_t fp_add(int16_t x, int16_t y) {
    return x + y;
}
int16_t fp_sub(int16_t x, int16_t y) {
    return x - y;
}
int16_t fp_mul(int16_t x, int16_t y, uint16_t precision) {
    int32_t mul = ((int32_t) x) * ((int32_t) y);
    return (int16_t) (mul >> precision);
}
int16_t fp_div(int16_t x, int16_t y, uint16_t precision) {
    int32_t xLarge = ((int32_t) x) << precision;
    return (int16_t) (xLarge / y);
}
int16_t fp_neg(int16_t x) {
    return -1 * x;
}
int16_t fp_mod(int16_t x, int16_t m, uint16_t precision) {
    int16_t div = fp_div(x, m, precision);
    int16_t floorDiv = div & ~((1 << precision) - 1);
    return fp_add(x, fp_neg(fp_mul(floorDiv, m, precision)));
}
int16_t convert_fp(int16_t x, uint16_t old_precision, uint16_t new_precision) {
    return (x * (1 << new_precision)) / (1 << old_precision);
}
int16_t float_to_fp(float x, uint16_t precision) {
    return (int16_t) (x * (1 << precision));
}
int16_t int_to_fp(int16_t x, uint16_t precision) {
    return x * (1 << precision);
}
int16_t fp_round_to_int(int16_t x, uint16_t precision) {
    int8_t should_invert_sign = 0;
    if (x < 0) {
        should_invert_sign = 1;
        x = fp_neg(x);
    }
    int16_t fractionMask = (1 << precision) - 1;
    int16_t fractionalPart = x & fractionMask;
    int16_t integerPart = x & ~(fractionMask);
    int16_t roundedVal;
    int16_t one_half = 1 << (precision - 1);
    if (fractionalPart >= one_half) {
        roundedVal = fp_add(integerPart, int_to_fp(1, precision));
    } else {
        roundedVal = integerPart;
    }
    if (should_invert_sign) {
        return fp_neg(roundedVal);
    }
    return roundedVal;
}
int16_t fp_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    if (x >= 0) {
        return x;
    }
    return 0;
}
int16_t fp_leaky_relu(int16_t x, uint16_t precision) {
    UNUSED(precision);
    int16_t isPositive = (int16_t) (x > 0);
    int16_t leakyX = (x >> 2);
    return isPositive * x + (1 - isPositive) * leakyX;
}
int16_t fp_linear(int16_t x, uint16_t precision) {
    UNUSED(precision);
    return x;
}
int16_t fp_tanh(int16_t x, uint16_t precision) {
    int16_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }
    int16_t one_fourth = 1 << (precision - 2);
    int16_t one_half = 1 << (precision - 1);
    int16_t one = int_to_fp(1, precision);
    int16_t half_x = fp_mul(x, one_half, precision);
    int16_t quarter_x = fp_mul(x, one_fourth, precision);
    int16_t numerator = fp_add(one_half, fp_mul(quarter_x, quarter_x, precision));
    int16_t denominator = fp_add(one_half, fp_mul(half_x, half_x, precision));
    int16_t rational_factor = fp_div(numerator, denominator, precision);
    int16_t result = fp_mul(x, rational_factor, precision);
    if (should_invert_sign) {
        result = fp_neg(result);
    }
    int16_t neg_one = fp_neg(one);
    if (result > one) {
        return one;    
    } else if (result < neg_one) {
        return neg_one;
    }
    return result;
}
int16_t fp_sigmoid(int16_t x, uint16_t precision) {
    uint8_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }
    int16_t one = 1 << precision;
    int16_t one_half = 1 << (precision - 1);
    int16_t half_x = fp_mul(x, one_half, precision);
    int16_t tanh = fp_tanh(half_x, precision);
    int16_t result = fp_mul(fp_add(tanh, one), one_half, precision);
    if (should_invert_sign) {
        result = one - result;
    }
    return result;
}
int32_t fp32_add(int32_t x, int32_t y) {
    return x + y;
}
int32_t fp32_neg(int32_t x) {
    return -1 * x;
}
int32_t fp32_sub(int32_t x, int32_t y) {
    return fp32_add(x, fp32_neg(y));
}
int32_t fp32_mul(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = (int64_t) x;
    int64_t yLarge = (int64_t) y;

    return (int32_t) ((xLarge * yLarge) >> precision);
}
int32_t fp32_div(int32_t x, int32_t y, uint16_t precision) {
    int64_t xLarge = ((int64_t) x) << precision;
    return (int32_t) (xLarge / y);
}
int32_t fp32_sqrt(int32_t x, uint16_t precision) {
    if (x < 0) {
        return 0;
    }
    int32_t hundred = int_to_fp32(100, precision);
    volatile int32_t a = x;
    volatile int32_t n = 1;
    while (a > hundred) {
        a /= 100;
        n *= 10;
    }

    const int32_t one256 = 1 << (precision - 8);
    const int32_t one32 = 1 << (precision - 5);
    const int32_t oneEight = 1 << (precision - 3);
    const int32_t oneHalf = 1 << (precision - 1);
    const int32_t two = 1 << (precision + 1);
    const int32_t eight = 1 << (precision + 3);
    const int32_t thirtyTwo = ((int32_t) 1) << (precision + 5);

    volatile int32_t sqrtPart = 0;

    if (a <= one256) {
        sqrtPart = fp32_add(a << 4, 1 << (precision - 6));  // 16x + 1/64
    } else if (a <= one32) {
        sqrtPart = fp32_add(a << 2, 1 << (precision - 5));  // 4x + 1/16
    } else if (a <= oneEight) {
        sqrtPart = fp32_add(a << 1, 1 << (precision - 3));  // 2x + 1/8
    } else if (a <= oneHalf) {
        sqrtPart = fp32_add(a, 1 << (precision - 2));  // x + 1/4
    } else if (a <= two) {
        sqrtPart = fp32_add(a >> 1, (int32_t) oneHalf);  // x/2 + 1/2
    } else if (a <= eight) {
        sqrtPart = fp32_add(a >> 2, 1 << precision);  // x/4 + 1
    } else if (a <= thirtyTwo) {
        sqrtPart = fp32_add(a >> 3, 1 << (precision + 1));  // x/8  + 2
    } else {
        sqrtPart = fp32_add(a >> 4, 1 << (precision + 2));  // x/16 + 4
    }

    return sqrtPart * n;
    // return fp32_mul(sqrtPart, n, precision);
}
int32_t int_to_fp32(int32_t x, uint16_t precision) {
    return ((int32_t) x) << precision;
}

static uint16_t SPARSEMAX_BUFFER[NUM_OUTPUTS] = {0};


matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2) {
    // if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
    //     return NULL_PTR;
    // }
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;
    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;
    uint16_t i, j;
    for (i = rows; i > 0; i--) {
        rowOffset = i - 1;
        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_add(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset]);
        }
    }
    return result;
}

matrix *matrix_neg(matrix *result, matrix *mat, uint16_t precision) {
    return scalar_product(result, mat, int_to_fp(-1, precision), precision);
}

matrix *matrix_multiply_vanilla(matrix *result, matrix *mat1, matrix *mat2, uint16_t precision) {
    // if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t n = mat1->numRows;
    uint16_t m = mat1->numCols;
    uint16_t p = mat2->numCols;
    uint16_t i, j, k;
    uint16_t outerRow, innerRow, resultRow;
    int16_t sum, prod;
    for (i = n; i > 0; i--) {
        outerRow = (i - 1) * m;
        for (j = p; j > 0; j--) {
            sum = 0;
            for (k = m; k > 0; k--) {
                innerRow = (k - 1) * p;
                prod = fp_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                sum = fp_add(sum, prod);
            }
            resultRow = (i - 1) * p;
            result->data[resultRow + (j - 1)] = sum;
        }
    }
    return result;
}

int16_t dot_product(matrix *vec1, matrix *vec2, uint16_t precision) {
    uint16_t i;
    uint16_t vec1Idx, vec2Idx;
    int16_t result = 0;
    for (i = vec1->numCols; i > 0; i--) {
        vec1Idx = i - 1;
        vec2Idx = vec2->numCols * (i - 1);
        result = fp_add(result, fp_mul(vec1->data[vec1Idx], vec2->data[vec2Idx], precision));
    }
    return result;
}



matrix *matrix_hadamard(matrix* result, matrix *mat1, matrix *mat2, uint16_t precision) {
    // if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
    //     return NULL_PTR;
    // }
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;
    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;
    uint16_t i, j;
    for (i = rows; i > 0; i--) {
        rowOffset = i - 1;
        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;
        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_mul(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset], precision);
        }
    }
    return result;
}


matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, uint16_t precision) {
    // if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_mul(mat->data[i - 1], scalar, precision);
    }
    return result;
}


matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar) {
    // if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_add(mat->data[i - 1], scalar);
    }
    return result;
}


matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision) {
    // if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = (*fn)(mat->data[i - 1], precision);
    }
    return result;
}


matrix *matrix_replace(matrix *dst, matrix *src) {
    // if ((dst->numRows != src->numRows) || (dst->numCols != src->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t i;
    for (i = dst->numRows * dst->numCols; i > 0; i--) {
        dst->data[i - 1] = src->data[i - 1];
    }
    return dst;
}


matrix *matrix_set(matrix *mat, int16_t value) {
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        mat->data[i - 1] = value;
    }
    return mat;
}


int16_t matrix_sum(matrix *mat) {
    int16_t sum = 0;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        sum = fp_add(mat->data[i - 1], sum);
    }
    return sum;
}


int16_t matrix_min(matrix *mat) {
    int16_t min_value = 32767;  // 2^15 - 1
    int16_t val;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        val = mat->data[i - 1];
        if (val < min_value) {
            min_value = val;
        }
    }
    return min_value;
}



matrix *vstack(matrix *result, matrix *mat1, matrix *mat2) {
    // if ((mat1->numRows + mat2->numRows != result->numRows) || (mat1->numCols != mat2->numCols) ||
    //     (mat1->numCols != result->numCols)) {
    //     return NULL_PTR;
    // }
    uint16_t cols = mat1->numCols;
    uint16_t i;
    for (i = mat1->numRows * cols; i > 0; i--) {
        result->data[i-1] = mat1->data[i-1];
    }
    uint16_t offset = mat1->numRows * cols;
    for (i = mat2->numRows * cols; i > 0; i--) {
        result->data[offset + i - 1] = mat2->data[i-1];
    }
    return result;
}


int16_t argmax(matrix *vec) {
    if (vec->numRows <= 0) {
        return -1;
    }
    uint16_t numCols = vec->numCols;
    int16_t max = vec->data[0];
    int16_t max_index = 0;
    uint16_t i;
    int16_t val;
    for (i = vec->numRows - 1; i > 0; i--) {
        val = vec->data[i * numCols];
        if (val > max) {
            max_index = i;
            max = val;
        }
    }
    return max_index;
}


uint16_t *argsort(matrix *vec, uint16_t *result) {
    uint16_t i;
    for (i = 0; i < vec->numRows; i++) {
        result[i] = i * vec->numCols;
    }
    uint16_t j, k;
    uint16_t idx1, idx2;
    int16_t t;
    for (k = vec->numRows; k > 0; k--) {
        i = vec->numRows - k;

        for (j = i; j > 0; j--) {
            idx1 = result[j-1];
            idx2 = result[j];
            if (vec->data[idx2] > vec->data[idx1]) {
                t = result[j-1];
                result[j-1] = result[j];
                result[j] = t;
            }
        }
    }
    return result;
}


matrix *sparsemax(matrix *result, matrix *vec, uint16_t precision) {
    uint16_t *sortedIndices = SPARSEMAX_BUFFER;
    argsort(vec, sortedIndices);
    int16_t partialSum = 0;
    int16_t one = 1 << precision;
    int16_t zk = 0;
    int16_t coordinate = 0;
    uint16_t k = 0;
    uint16_t i;
    for (i = vec->numRows; i > 0; i--) {
        k = vec->numRows - i + 1;
        zk = vec->data[sortedIndices[k - 1]];
        partialSum = fp_add(partialSum, zk);
        coordinate = fp_add(one, fp_mul(int_to_fp(k, precision), zk, precision));
        if (coordinate <= partialSum) {
            k = k - 1;
            partialSum = fp_add(partialSum, fp_neg(zk));
            break;
        }
    }
    int16_t kz = int_to_fp(k, precision);
    int16_t sumMinusOne = fp_add(partialSum, fp_neg(one));
    int16_t threshold = fp_div(sumMinusOne, kz, precision);
    uint16_t j, idx;
    int16_t diff;
    for (j = vec->numRows; j > 0; j--) {
        idx = (j - 1) * vec->numCols;
        diff = fp_sub(vec->data[idx], threshold);
        result->data[idx] = fp_relu(diff, precision);
    }
    return result;
}

static dtype PADDING_BUFFER[PADDING_BUFFER_LENGTH] = {0};

static dtype FILTER_BUFFER[FILTER_BUFFER_LENGTH] = {0};


matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision) {
    /**
     * Implementation of a dense feed-forward layer using matrix operations.
     */
    result = matrix_multiply_vanilla(result, W, input, precision);

    // Only add bias if given 
    if (b != NULL_PTR) {
        result = matrix_add(result, result, b);
    }

    result = apply_elementwise(result, result, activation, precision);
    return result;
}

matrix *maxpooling(matrix* result, matrix *input, uint16_t pool_numRows, uint16_t pool_numCols){
    /**
     * Implementation of maxpooling layer
     */
    uint16_t result_numRows = input->numRows / pool_numRows;
    uint16_t result_numCols = input->numCols / pool_numCols;
    uint16_t i, j, x, y, kx, ky, input_offset, result_offset;
    int16_t max;
    for (i = 0; i < result_numRows; i ++){
        for (j = 0; j < result_numCols; j ++){

            // (i, j) is the coordinate of each element after maxpooling
            // (x, y) is the coordinate of top-left element among all corresponding points in the original input matrix

            x = i * pool_numRows;
            y = j * pool_numCols;

            max = -32768;
            for (kx = 0; kx < pool_numRows; kx ++){
                for (ky = 0; ky < pool_numCols; ky ++){
                    // traverse the entire sub-block that are related to this pooling
                    input_offset = (x + kx) * input->numCols + (y + ky);
                    if (max < input->data[input_offset]){
                        max = input->data[input_offset];  // if a bigger number found, update max
                    }

                }
            }
            result_offset = i * result_numCols + j;
            result->data[result_offset] = max;

        }
    }
    return result;
}


matrix *maxpooling_filters(matrix *result, matrix *input, uint16_t numFilters, uint16_t pool_numRows, uint16_t pool_numCols){
    /**
     * Iteration for each filter
     * one conv2d layer usually has multiple filters, we do maxpooling one by one
     */

    uint16_t i, filter_offset, result_offset, filter_length = input->numRows * input->numCols, result_length = result->numRows * result->numCols;
    int16_t *filterData = input->data, *resultData = result->data;

    for (i = numFilters; i > 0; i --){
        filter_offset = (i - 1) * filter_length;
        result_offset = (i - 1) * result_length;

        input->data = filterData + filter_offset;
        result->data = resultData + result_offset;

        /* process one filter at a time */
        maxpooling(result, input, pool_numRows, pool_numCols);
    }
    return result;
}

matrix *flatten(matrix* result, matrix *input, uint16_t num_filter){
    /**
     * Implementation of flatten layer for CNN
     * the result of conv2d_maxpooling or conv2d_filter is saved in the order of filter by filter
     * however, the flatten result should be in this following way according to Tensorflow
     * f0[0], f1[0], ..., fn[0], f0[1], f1[1], ..., fn[1], ..., f0[n], f1[n], fn[n]
     */
    uint16_t i, j, input_offset, result_offset = 0;
    uint16_t filter_length;
    filter_length = input->numCols * input->numRows;
    for (i = 0; i < filter_length; i ++ ){
        for (j = 0; j < num_filter; j ++){
            input_offset = i + j * filter_length;   // get the ith element of the jth filter
            result->data[result_offset++] = input->data[input_offset];   // append it to result
            result->data[result_offset++] = 0; // for LEA, we have to append 0 to each number
        }
    }
    return result;
}

matrix *padding_same(matrix *result, matrix *input, matrix *filter, uint16_t stride_numRows, uint16_t stride_numCols) {
    uint16_t input_numRows = input->numRows, input_numCols = input->numCols, filter_numRows = filter->numRows, filter_numCols = filter->numCols;
    uint16_t pad_along_numRows, pad_along_numCols, i, input_offset, padding_offset;
    if (input_numRows % stride_numRows) {
        pad_along_numRows = filter_numRows - input_numRows % stride_numRows;
    }
    else {
        pad_along_numRows = filter_numRows - stride_numRows;
    }
    if (input_numCols % stride_numCols) {
        pad_along_numCols = filter_numCols - input_numCols % stride_numCols;
    }
    else {
        pad_along_numCols = filter_numCols - stride_numCols;
    }

    result->numRows = input->numRows + pad_along_numRows;
    result->numCols = input->numCols + pad_along_numCols;

    memset(PADDING_BUFFER, 0, result->numRows * result->numCols * sizeof(dtype));

    for (i = 0; i < input_numRows; i ++) {
        input_offset = i * input_numCols;
        padding_offset = ((pad_along_numRows >> 1) + i) * result->numCols + (pad_along_numCols >> 1);
        memcpy(PADDING_BUFFER + padding_offset, input->data + input_offset, input_numCols << 1);
    }

    result->data = PADDING_BUFFER;
    return result;
}

matrix *filter_simple(matrix *result, matrix *input, matrix *filter, uint16_t precision, uint16_t stride_numRows, uint16_t stride_numCols){
    /**
     * Implementation of one filter of a conv2d layer
     */
    uint16_t input_numRows = input->numRows;
    uint16_t input_numCols = input->numCols;
    uint16_t filter_numRows = filter->numRows;
    uint16_t filter_numCols = filter->numCols;
    uint16_t i, j, m, n, input_offset, filter_offset, result_offset = 0;
    int16_t mult_result, sum = 0, mult1, mult2;

    for (i = 0; i <= input_numRows - filter_numRows; i += stride_numRows){
        for (j = 0; j <= input_numCols - filter_numCols; j += stride_numCols){
            // (i,j) is the coordinate of the top-left element of the moving filter
            sum = 0;
            for (m = i; m < i + filter_numRows; m ++){
                for (n = j; n < j + filter_numCols; n ++){  // calculate element-wise matrix product between the filter and corresponding section in the input image
                    input_offset = m * input_numRows + n;
                    filter_offset = (m - i) * filter_numCols + (n - j);
                    mult1 = input->data[input_offset];
                    mult2 = filter->data[filter_offset];
                    mult_result = fp_mul(mult1, mult2, precision);
                    sum += mult_result;  // ATTENTION *** potential overflow issue ***
                }
            }
            result->data[result_offset ++] = sum;  // add bias
        }
    }
    return result;
}


matrix *filters_sum(matrix *result, matrix *input, matrix *filter, uint16_t numChannels, int16_t b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision, uint16_t stride_numRows, uint16_t stride_numCols, uint16_t padding, uint16_t conv_numRows, uint16_t conv_numCols){
    int16_t *filter_head = filter->data;
    int16_t *input_head = input->data;
    uint16_t i, result_length = result->numRows * result->numCols, input_length = input->numRows * input->numCols, filter_length = filter->numRows * filter->numCols, input_numRows = input->numRows, input_numCols = input->numCols;
    matrix temp = {FILTER_BUFFER, result->numRows, result->numCols};
    memset(result->data, 0, result_length * sizeof(dtype));

    for (i = numChannels; i > 0; i --){
        input->data = input_head + input_length * (i - 1);
        filter->data = filter_head + filter_length * (i - 1);
        if (padding == 1) {
            padding_same(input, input, filter, stride_numRows, stride_numCols);
        }
        filter_simple(&temp, input, filter, precision, stride_numRows, stride_numCols);
        matrix_add(result, result, &temp);
        input->numRows = input_numRows;
        input->numCols = input_numCols;
        input->data = input_head;
    }
    for (i = result_length; i > 0; i --){
        result->data[i - 1] = result->data[i - 1] + b;
    }
    result = apply_elementwise(result, result, activation, precision);
    return result;
}

matrix *conv2d(matrix *result, matrix *input, matrix *filter, uint16_t numFilters, uint16_t numChannels, int16_t *b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision, uint16_t stride_numRows, uint16_t stride_numCols, uint16_t padding){
    uint16_t i, result_length = result->numRows * result->numCols, filter_length = filter->numRows * filter->numCols * numChannels;
    int16_t *filter_head = filter->data, *result_head = result->data;
    uint16_t conv_numRows = (input->numRows - filter->numRows) / stride_numRows + 1;
    uint16_t conv_numCols = (input->numCols - filter->numCols) / stride_numCols + 1;
    for (i = numFilters; i > 0; i --){
        filter->data = filter_head + (i - 1) * filter_length;
        result->data = result_head + (i - 1) * result_length;
        filters_sum(result, input, filter, numChannels, b[i - 1], activation, precision, stride_numRows, stride_numCols, padding, conv_numRows, conv_numCols);
    }
    return result;
}

matrix *apply_leakyrelu(matrix *result, matrix *input, uint16_t precision){
    result = apply_elementwise(result, input, &fp_leaky_relu, precision);
    return result;
}


static int16_t MODEL_ARRAY_OUTPUT[MODEL_ARRAY_OUTPUT_LENGTH] = {0};

static int16_t MODEL_ARRAY_TEMP[MODEL_ARRAY_TEMP_LENGTH] = {0};

matrix *apply_model(matrix *output, matrix *input){

    uint32_t i = 0;
    int16_t *array = MODEL_ARRAY;
    int16_t *bias_array;

    uint16_t array_length = MODEL_ARRAY_LENGTH;

    uint16_t layer_class, activation, numChannels, filter_numRows, filter_numCols, stride_numRows, stride_numCols, filters_length, padding;
    uint16_t numFilters;
    output->data = MODEL_ARRAY_OUTPUT;

    i = 0;
    // Sequential model
    if (array[i] == 0){  // 1st element of the array tells the model type
        i ++;
        
        while (i != array_length){
            // next element of the array tells the layer class

            Serial.println(array[i]);  // for layer-wise overhead measurement

            /* layer class 0 - DENSE */
            if (array[i] == DENSE_LAYER){
                numFilters = 1;

                digitalWrite(signal_pin, HIGH);  // for layer-wise overhead measurement
                
                // extract and prepare layer parameters
                layer_class = array[i];
                activation = array[i+1];
                uint16_t kernel_numRows = array[i+2];
                uint16_t kernel_numCols = array[i+3];
                uint16_t bias_numRows = array[i+4];
                uint16_t bias_numCols = array[i+5];
                i += 6;
                uint16_t kernel_length = kernel_numRows * kernel_numCols;
                uint16_t bias_length = bias_numRows * bias_numCols;

                // extract layer weights
                int16_t *kernel_array = &array[i];
                i += kernel_length;
                bias_array = &array[i];
                i += bias_length;

                // prepare output
                uint16_t output_numRows = kernel_numRows;
                uint16_t output_numCols = input->numCols;
                output->numRows = output_numRows;
                output->numCols = output_numCols;

                // initialize weight matrix
                matrix kernel = {kernel_array, kernel_numRows, kernel_numCols};
                matrix bias = {bias_array, bias_numRows, bias_numCols};

                // execute dense layer
                if (activation == RELU_ACTIVATION){
                    dense(output, input, &kernel, &bias, &fp_relu, FIXED_POINT_PRECISION);
                }
                else if (activation == SIGMOID_ACTIVATION){
                    dense(output, input, &kernel, &bias, &fp_sigmoid, FIXED_POINT_PRECISION);
                }
                else{
                    dense(output, input, &kernel, &bias, &fp_linear, FIXED_POINT_PRECISION);
                }

                digitalWrite(signal_pin, LOW); // for layer-wise overhead measurement
                delay(100);                    // for layer-wise overhead measurement

            }

            /* layer class 1 - LeakyReLU */
            else if (array[i] == LEAKY_RELU_LAYER){
                output->numRows = input->numRows;
                output->numCols = input->numCols;
                apply_leakyrelu(output, input, FIXED_POINT_PRECISION);
                i ++;
            }

            /* layer class 2 - Conv2D */
            else if (array[i] == CONV2D_LAYER){

                digitalWrite(signal_pin, HIGH);  // for layer-wise overhead measurement

                // extract and prepare layer parameters
                layer_class = array[i];
                activation = array[i+1];
                numFilters = array[i+2];
                numChannels = array[i+3];
                filter_numRows = array[i+4];
                filter_numCols = array[i+5];
                stride_numRows = array[i+6];
                stride_numCols = array[i+7];
                filters_length = array[i+8];
                padding = array[i+9];
                i += 10;

                // prepare output
                if (padding == 1){
                    output->numRows = input->numRows / stride_numRows;
                    if (input->numRows % stride_numRows > 0){
                        output->numRows ++;
                    }
                    output->numCols = input->numCols / stride_numCols;
                    if (input->numCols % stride_numRows > 0){
                        output->numCols ++;
                    }
                }
                else {
                    output->numRows = (input->numRows - filter_numRows + 1) / stride_numRows;
                    if ((input->numRows - filter_numRows + 1) % stride_numRows > 0){
                        output->numRows ++;
                    }
                    output->numCols = (input->numCols - filter_numCols + 1) / stride_numCols;
                    if ((input->numCols - filter_numCols + 1) % stride_numCols > 0){
                        output->numCols ++;
                    }
                }

                // extract and prepare weights
                int16_t *filters_array = array + i;
                matrix filters = {filters_array, filter_numRows, filter_numCols};
                i += filters_length;

                bias_array = array + i;
                i += numFilters;


                // execute conv2d layer
                if (activation == RELU_ACTIVATION){
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_relu, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }
                else if (activation == SIGMOID_ACTIVATION){
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_sigmoid, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }
                else{
                    conv2d(output, input, &filters, numFilters, numChannels, bias_array, &fp_linear, FIXED_POINT_PRECISION, stride_numRows, stride_numCols, padding);
                }

            }

            /* layer class 3 - MaxPooling2D */
            else if (array[i] == MAXPOOLING2D_LAYER){
                uint16_t pool_numRows = array[i+1];
                uint16_t pool_numCols = array[i+2];
                stride_numRows = array[i+3];
                stride_numCols = array[i+4];
                padding = array[i+5];
                i += 6;

                output->numRows = input->numRows / pool_numRows;
                output->numCols = input->numCols / pool_numCols;

                maxpooling_filters(output, input, numFilters, pool_numRows, pool_numCols);

                digitalWrite(signal_pin, LOW); // for layer-wise overhead measurement
                delay(100);                    // for layer-wise overhead measurement
            }

            /* layer class 4 - Conv2D Flatten */
            else if (array[i] == FLATTEN_LAYER){

                digitalWrite(signal_pin, HIGH);  // for layer-wise overhead measurement

                i += 1;
                output->numRows = input->numRows * input->numCols * numFilters;
                output->numCols = LEA_RESERVED;
                flatten(output, input, numFilters);
                numFilters = 1;
            }
            /* SKIP FOR INFERENCE TIME IMPLEMENTATION - layer class 5 - Dropout Layer */
            else if (array[i] == DROPOUT_LAYER){
                i += 1;
                numFilters = 1;
            }

            /* copy output matrix and reference input to copied output */
            memcpy(MODEL_ARRAY_TEMP, output->data, output->numRows * output->numCols * numFilters << 1);
            input->data = MODEL_ARRAY_TEMP;
            input->numRows = output->numRows;
            input->numCols = output->numCols;

        }
    }

    return output;
}

// main
uint16_t inference(void){

    inputFeatures.numRows = INPUT_NUM_ROWS;
    inputFeatures.numCols = INPUT_NUM_COLS;
    inputFeatures.data = input_buffer;

    outputLabels.numRows = OUTPUT_NUM_LABELS;
    outputLabels.numCols = LEA_RESERVED;
    outputLabels.data = output_buffer;

    apply_model(&outputLabels, &inputFeatures);
    label = argmax(&outputLabels);

    return label;
}

void setup() {
    Serial.begin(9600); 
    pinMode(signal_pin, OUTPUT);
}

void loop(){
    inference();
    // Serial.println(iteration++);
    // Serial.print("inference result:\n");
    // Serial.println(inference());
    Serial.print("\n\n\n");
    delay(3000);

}