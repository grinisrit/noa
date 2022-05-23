auto matrixElements = [=] __cuda_callable__ (
    Index rows,
    Index columns,
    Index row,
    Index localIdx,
    Index& columnIdx,
    Real& value );

