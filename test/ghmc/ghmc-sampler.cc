#include "routines.hh"

int main()
{
    if(!sample_normal_distribution("ghmc_sample_normal_distribution.pt")) return 1;

    if(!sample_funnel_distribution("ghmc_sample_funnel_distribution.pt")) return 1;

    return 0;
}

