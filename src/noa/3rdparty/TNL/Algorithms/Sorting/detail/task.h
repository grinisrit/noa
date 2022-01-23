// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Xuan Thang Nguyen

#pragma once

#include <iostream>

#include <noa/3rdparty/TNL/Cuda/CudaCallable.h>

namespace noa::TNL {
    namespace Algorithms {
        namespace Sorting {

struct TASK
{
    //start and end position of array to read and write from
    int partitionBegin, partitionEnd;
    //-----------------------------------------------
    //helper variables for blocks working on this task

    int iteration;
    int pivotIdx;
    int dstBegin, dstEnd;
    int firstBlock, blockCount;//for workers read only values

    __cuda_callable__
    TASK(int begin, int end, int iteration)
        : partitionBegin(begin), partitionEnd(end),
        iteration(iteration), pivotIdx(-1),
        dstBegin(-151561), dstEnd(-151561),
        firstBlock(-100), blockCount(-100)
        {}

    __cuda_callable__
    void initTask(int firstBlock, int blocks, int pivotIdx)
    {
        dstBegin= 0; dstEnd = partitionEnd - partitionBegin;
        this->firstBlock = firstBlock;
        blockCount = blocks;
        this->pivotIdx = pivotIdx;
    }

    __cuda_callable__
    int getSize() const
    {
        return partitionEnd - partitionBegin;
    }

    TASK() = default;
};

inline std::ostream& operator<<(std::ostream & out, const TASK & task)
{
    out << "[ ";
    out << task.partitionBegin << " - " << task.partitionEnd;
    out << " | " << "iteration: " << task.iteration;
    out << " | " << "pivotIdx: " << task.pivotIdx;
    return out << " ] ";
}

        } // namespace Sorting
    } // namespace Algorithms
} // namespace noa::TNL
