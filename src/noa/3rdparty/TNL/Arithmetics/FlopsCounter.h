// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {

class tnlFlopsCounter
{
	bool compute_flops;

	long int adding_flops;

	long int multiplying_flops;

	long int dividing_flops;

	long int functions_flops;

	public:

	tnlFlopsCounter()
	: compute_flops( true ),
      adding_flops( 0 ),
      multiplying_flops( 0 ),
      dividing_flops( 0 )
	{};

	void recordAdding( const int ops = 1 )
	{
	   if( compute_flops ) adding_flops += ops;
	};

	void recordMultiplying( const int ops = 1 )
	{
		if( compute_flops ) multiplying_flops += ops;
	};

	void recordDividing( const int ops = 1 )
	{
		if( compute_flops ) dividing_flops += ops;
	};

       void recordFunction( const int ops = 1 )
       {
          if( compute_flops ) functions_flops += ops;
       };


	void turnFlopsCountingOn()
	{
		compute_flops = true;
	};

	void turnFlopsCountingOff()
	{
		compute_flops = false;
	};

	long int getFlops()
	{
		return adding_flops + multiplying_flops + dividing_flops + functions_flops;
	};

	void resetFlops()
	{
		adding_flops = 0;
		multiplying_flops = 0;
	   dividing_flops = 0;
	   functions_flops = 0;
	};
};

extern tnlFlopsCounter tnl_flops_counter;

} // namespace noa::TNL
