#include <benchmark/benchmark.h>
#include <torch/torch.h>


static void BM_TorchSave(benchmark::State & state)
{
    torch::manual_seed(987654);
    auto tensor = torch::randn({2000000, 4});
    for (auto _ : state)
        torch::save(tensor, "tensor.pt");
}
BENCHMARK(BM_TorchSave);

static void BM_TorchLoad(benchmark::State & state)
{
    torch::manual_seed(987654);
    auto tensor = torch::randn({2000000, 4});
    torch::save(tensor, "tensor.pt");
    for (auto _ : state)
        torch::load(tensor, "tensor.pt");
}
BENCHMARK(BM_TorchLoad);