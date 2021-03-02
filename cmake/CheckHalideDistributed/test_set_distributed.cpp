#include <vector>
#include <Halide.h>

int main(void) {
    std::vector<int> sizes = { 32 };
    Halide::Runtime::Buffer<int, 1> buf(sizes);
    buf.set_distributed(sizes);
    return 0;
}
