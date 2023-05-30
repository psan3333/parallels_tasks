#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include "cublas_v2.h"
#include "npy.h"
#include <chrono>
#include <iomanip>

#define CUDACHECK(status, name) if (status != cudaSuccess) { throw std::runtime_error(name); } 
#define CUBLASCHECK(status, name) if (status != CUBLAS_STATUS_SUCCESS) { throw std::runtime_error(name); } 

uint32_t NOD(uint32_t a, uint32_t b)
{
    while (a > 0 && b > 0) {
        if (a > b) {
            a %= b;
        }
        else {
            b %= a;
        }
    }
    return a + b;
}

template <class T>
class cudarray
{
private:

    T* _data = NULL;  // buffer for typed data
    uint32_t _size = 0;  // size of buffer

public:
    // clear buffer if there was something and allocates new memory block
    void set_size(uint32_t size)
    {
        if (this->_data) 
            // we need to clear array if it was filled with something at first
            CUDACHECK(cudaFree(this->_data), "free cudarray data in reallocation section")

        CUDACHECK(cudaMalloc((void**)&this->_data, sizeof(T) * size), "alloc cudarray")
        this->_size = size;
    }

    // copies vector into class object
    cudarray& operator=(const std::vector<T> arr)
    {
        if (arr.size() != this->_size)
            // resize to fit new data
            this->set_size(arr.size());

        CUDACHECK(cudaMemcpy(this->_data, arr.data(), this->_size * sizeof(T), cudaMemcpyHostToDevice), "copy arr to cudarray")
        return *this;
    }

    // destructor
    ~cudarray()
    {
        if (!this->_data) {
            CUDACHECK(cudaFree(this->_data), "cudarray destructor")
        }
    }

    // returns buffer size
    uint32_t size()
    {
        return this->_size;
    }

    // returns reference to buffer
    T* data()
    {
        return this->_data;
    }
};


// interface for linear layers
template <class T>
class Layer
{
protected:
    cudarray<T> in_x;
    cudarray<T> out_x;
    cudarray<T> grad;
    cudarray<T> err_x;
    dim3 threads = { 1,1,1 };
    dim3 blocks = { 1,1,1 };
public:
    // method for forward propagation in neural network
    virtual cudarray<T> forward(cudarray<T> x) = 0;
    // method for reading weights from binary file with them
    virtual void read_weights(std::string filepath) {};
};

// FC layer
class Linear : public Layer<float>
{
private:
    cudarray<float> weights;
    cudarray<float> buff;
    cudarray<float> bias;
    cublasHandle_t handle;
    bool h_bias;
    float alpha = 1, beta;

public:

    Linear (uint32_t in, uint32_t out, bool has_bias = true) : h_bias(has_bias)
    {
        this->weights.set_size(in * out);
        this->grad.set_size(in * out);
        this->err_x.set_size(in);
        this->out_x.set_size(out);
        this->in_x.set_size(in);

        this->beta = has_bias;
        if (has_bias)
        {
            this->bias.set_size(out);
        }

        this->threads.x = NOD(out, 1024);
        this->blocks.x = out / this->threads.x;
        CUBLASCHECK(cublasCreate(&this->handle), "handle creation in Linear");
    }

    ~Linear()
    {
        CUBLASCHECK(cublasDestroy(this->handle), "Linear destructor - cublasDestroy handle")
    }

    cudarray<float> forward(cudarray<float> x)
    {
        if (this->in_x.size() != x.size())
            throw std::runtime_error("Invalid input size - Linear forward prop");
        this->in_x = x;
        if (this->h_bias) {
            CUDACHECK(
                cudaMemcpy(this->out_x.data(), this->bias.data(), sizeof(float) * this->out_x.size(), cudaMemcpyDeviceToDevice),
                "copy biases input output section - Linear forward prop"
            )
        }

        CUBLASCHECK(cublasSgemm(
            this->handle, 
            CUBLAS_OP_N, 
            CUBLAS_OP_N, 
            1, 
            this->out_x.size(), 
            this->in_x.size(), 
            &this->alpha, 
            this->in_x.data(), 
            1, 
            this->weights.data(), 
            this->in_x.size(), 
            &this->beta, 
            this->out_x.data(), 
            1), "forward prop Linear - cublasSgemm")
        return this->out_x;
    }
    
    void read_weights(std::string filepath)
    {
        std::vector<npy::ndarray_len_t> shape, bias_shape;
        std::vector<float> data, bias_data;
        npy::LoadArrayFromNumpy<float>(filepath, shape, data);

        if (shape[1] != this->in_x.size() || shape[0] != this->out_x.size())
            throw std::runtime_error("Not a valid shape");

        CUDACHECK(
            cudaMemcpy(this->weights.data(), data.data(), this->weights.size() * sizeof(float), cudaMemcpyHostToDevice), 
            "copy weights from 'filepath' file - Linear read_weights"
        )

        if (this->h_bias)
        {
            npy::LoadArrayFromNumpy<float>("bias_" + filepath, bias_shape, bias_data);
            CUDACHECK(
                cudaMemcpy(this->bias.data(), bias_data.data(), this->out_x.size() * sizeof(float), cudaMemcpyHostToDevice),
                "copy biases from 'filepath' file - Linear read_weights"
            )
        }
    }

};

__global__ void sigmoid_vectorized(float* in, float* out, int size)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) out[id] = 1 / (1 + exp(-in[id]));
}

// sigmoid activation layer
class Sigmoid : public Layer<float>
{
public:
    Sigmoid(uint32_t size) {
        this->out_x.set_size(size);
        this->err_x.set_size(size);
        this->in_x.set_size(size);
        this->threads.x = NOD(size, 1024);
        this->blocks.x = size / this->threads.x;
    }

    cudarray<float> forward(cudarray<float> x)
    {
        if (this->in_x.size() != x.size())
            throw std::runtime_error("Not a valid size");
        this->in_x = x;

        sigmoid_vectorized<<<this->blocks, this->threads>>>(this->in_x.data(), this->out_x.data(), this->in_x.size());

        return this->out_x;
    }

    // throws exception at calling
    void read_weights(std::string filepath) {}
};

// class which compresses different layers into one neural network
template <class T>
class Model
{
private:
    std::vector<Layer<T>*> layers;

public:
    cudarray<T> forward(cudarray<T> x)
    {
        for (auto lay : this->layers)
            x = lay->forward(x);
        return x;
    }

    void push_back(Layer<T>* lay)
    {
        this->layers.push_back(lay);
    }

    void load_from_numpy(std::string filepath)
    {
        auto id = filepath.rfind('.');
        std::string filename, type;

        for (int i = 0; i < id; i++) {
            filename += filepath[i];
        }
        for (int i = id; i < filepath.size(); i++) {
            type += filepath[i];
        }
        for (int i = 0; i < layers.size(); i++) {
            	std::cout << std::string("./") + filename + std::to_string(i) + type << std::endl;
		this->layers[i]->read_weights(std::string("./") + filename + std::to_string(i) + type);
        }
    }

    ~Model()
    {
        for (auto lay : this->layers) {
            delete lay;
        }
    }
};
