#pragma once

#include <string>
#include <chrono>
#include <functional>
#include <iostream>
#include <utility>

struct TIMER
{
    std::function<void(double)> f;
    std::chrono::high_resolution_clock::time_point begin;

    TIMER( std::function< void( double ) > func =
              []( double res )
           {
              std::cout << res << std::endl;
           } )
    : f( std::move( func ) ), begin( std::chrono::high_resolution_clock::now() )
    {}

    ~TIMER()
    {
        auto end = std::chrono::high_resolution_clock::now();
        double result = (std::chrono::duration_cast<std::chrono::microseconds >(end - begin).count() / 1000.);
        f(result);
    }
};