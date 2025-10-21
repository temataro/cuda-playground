#include <stdio.h>

// clang-format off
#define RED     "\033[31m"
#define GRN     "\033[32m"
#define BLU     "\033[34m"
#define RST     "\033[0m"

#define STR_HELPER(x) #x
#define STR(x)        STR_HELPER(x)

#define ERR  RED "[ERROR]\t" RST
#define INFO GRN "[INFO]\t"  RST
// clang-format on

void get_device_properties()
{
    /*
     Returns a list of the device properties I'd care about
     from however many CUDA capable devices are on this machine.
     */
    int n_devices;

    cudaGetDeviceCount(&n_devices);
    fprintf(stderr, INFO "%d CUDA capable devices detected.\n", n_devices);

    for (int i = 0; i < n_devices; i++)
    {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        fprintf(stderr, "\t\t+" "----------------------+\n", i);
        fprintf(stderr, "\t\t|" INFO "Device #%d      |\n", i);
        fprintf(stderr, "\t\t+" "----------------------+\n\n", i);
        fprintf(stderr, INFO "Name:\t\t\t %s\n", p.name);
        fprintf(stderr, INFO "Warp Size:\t\t %d\n", p.warpSize);
        fprintf(stderr, INFO "Total Const Memory (KB): %zu\n", p.totalConstMem / 1024);
        fprintf(stderr, INFO "Total Glob Memory (MB):  %zu\n", p.totalGlobalMem / (1024 * 1024));
        fprintf(stderr, INFO "Max Threads per MP:\t %d\n", p.maxThreadsPerMultiProcessor);
        fprintf(stderr, INFO "Max Thread Dim:\t\t [%d, %d, %d]\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        fprintf(stderr, INFO "Max Grid Dim:\t\t [%d, %d, %d]\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        fprintf(stderr, INFO "Unified Addresssing:\t %d\n", p.unifiedAddressing);
    }
}
int main()
{
    get_device_properties();
    return 0;
}
