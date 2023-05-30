#include "layers.cuh"
#include <chrono>
#include <iomanip>

int main()
{
    try {
        Model<float> net;

        net.push_back(new Linear(32 * 32, 16 * 16, false));
        net.push_back(new Sigmoid(16 * 16));
        net.push_back(new Linear(16 * 16, 4 * 4, false));
        net.push_back(new Sigmoid(4 * 4));
        net.push_back(new Linear(4 * 4, 1, false));
        net.push_back(new Sigmoid(1));
        net.load_from_numpy("npy_weights.npy");

        cudarray<float> in;
        in.set_size(32 * 32);

        std::vector<npy::ndarray_len_t> shape;
        std::vector<float> in_data;
        npy::LoadArrayFromNumpy<float>("data.npy", shape, in_data);

        in = in_data;

        auto start = std::chrono::high_resolution_clock::now();

        cudarray<float> device_nn_output = net.forward(in);

        float nn_output = 0.0;
        CUDACHECK(
            cudaMemcpy(&nn_output, device_nn_output.data(), sizeof(float), cudaMemcpyDeviceToHost),
            "copy output from device to host"
        )

        auto finish = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count() << "ns\n";
        std::cout << std::fixed << std::setprecision(16) << nn_output << std::endl;
    }
    catch (std::exception& err) {
        std::cout << err.what() << std::endl;
        std::cout << "Programm execution ended with error" << std::endl;
        exit(-1);
    }

    return 0;
}
