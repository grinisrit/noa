#pragma once

extern int enzyme_allocated
         , enzyme_const
         , enzyme_dup
         , enzyme_duponneed
         , enzyme_out
         , enzyme_tape;

template <typename Retval, typename... Args>
Retval __enzyme_autodiff(Retval (*)(Args...), auto...);