# Warning
This fork is made to be used exclusively with [noa](https://github.com/grinisrit/noa).
Actually, the only part of it that concerns us is **src/TNL**.
Changes include: removed dependancy on `mpark::variant`, changes to header file paths _as they are in **noa**_.
This repo only exists to make the process of merging custom changes with upstream TNL easier.

## To NOA mantainers
To be used with NOA, TNL has to undergo some modifications, such as:
1. Replacing TNL header paths with their locations in NOA
2. Replacing `mpark::` and `experimental::` feature references with their standard C++17 implementation
3. Renaming `TNL` namespace to `noa::TNL`
4. Changing the locations of 3rdparty library headers to the ones used within NOA

For this purpose, `noa-ize.sh` script exists.
All changes that are made to TNL codebase should be automated (if possible) through this script to avoid doing these repetitive tasks manually every time.

# Original README

[![pipeline status](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/badges/develop/pipeline.svg)](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/commits/develop)

# Template Numerical Library

TNL is a collection of building blocks that facilitate the development of
efficient numerical solvers. It is implemented in C++ using modern programming
paradigms in order to provide flexible and user friendly interface. TNL provides
native support for modern hardware architectures such as multicore CPUs, GPUs,
and distributed systems, which can be managed via a unified interface.

Similarly to the STL, features provided by the TNL can be grouped into
several modules:

- _Core concepts_.
  The main concepts used in TNL are the _memory space_, which represents the
  part of memory where given data is allocated, and the _execution model_,
  which represents the way how given (typically parallel) algorithm is executed.
  For example, data can be allocated in the main system memory, in the GPU
  memory, or using the CUDA Unified Memory which can be accessed from the host
  as well as from the GPU. On the other hand, algorithms can be executed using
  either the host CPU or an accelerator (GPU), and for each there are many ways
  to manage parallel execution. The usage of memory spaces is abstracted with
  [allocators][allocators] and the execution model is represented by
  [devices][devices]. See the [Core concepts][core concepts] page for details.
- _[Containers][containers]_.
  TNL provides generic containers such as array, multidimensional array or array
  views, which abstract data management and execution of common operations on
  different hardware architectures.
- _Linear algebra._
  TNL provides generic data structures and algorithms for linear algebra, such
  as [vectors][vectors], [sparse matrices][matrices],
  [Krylov solvers][linear solvers] and [preconditioners][preconditioners].
   - Sparse matrix formats: CSR, Ellpack, Sliced Ellpack, Chunked Ellpack, Bisection Ellpack,
   - tridiagonal, multidiagonal
   - Lambda matrices (matrix elements are defined by C++ lambda functions)
   - Stationary solvers: Jacobi, SOR
   - Krylov solvers: CG, BiCGStab, BICGStab(l), GMRES, CWYGMRES, TFQMR
   - Preconditioners: Jacobi, ILU(0) (CPU only), ILUT (CPU only)
- _[Meshes][meshes]_.
  TNL provides data structures for the representation of structured or
  unstructured numerical meshes.
- _Solvers for differential equations._
  TNL provides a framework for the development of ODE or PDE solvers.
- _[Image processing][image processing]_.
  TNL provides structures for the representation of image data. Imports and
  exports from several file formats such as DICOM, PNG, and JPEG are provided
  using external libraries (see below).

See also [Comparison with other libraries](
https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/md_Pages_comparison_with_other_libraries.html).

[allocators]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Allocators.html
[devices]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Devices.html
[core concepts]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/md_Pages_core_concepts.html
[containers]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Containers.html
[vectors]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/classTNL_1_1Containers_1_1Vector.html
[matrices]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Matrices.html
[linear solvers]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Solvers_1_1Linear.html
[preconditioners]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Solvers_1_1Linear_1_1Preconditioners.html
[meshes]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Meshes.html
[image processing]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/namespaceTNL_1_1Images.html

TNL also provides several optional components:

- TNL header files in the
  [src/TNL](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/TNL)
  directory.
- Various pre-processing and post-processing tools in the
  [src/Tools](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Tools)
  directory.
- Python bindings and scripts in the
  [src/Python](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Python)
  directory.
- Examples of various numerical solvers in the
  [src/Examples](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Examples)
  directory.
- Benchmarks in the
  [src/Benchmarks](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/tree/develop/src/Benchmarks)
  directory.

These components can be individually enabled or disabled and installed by a
convenient `install` script. See the [Installation][installation] section in
the documentation for details.

## Documentation

See the [full documentation][full documentation] for information about:

- [installation instructions][installation],
- [usage hints][usage],
- [tutorials][tutorials],
- [API reference manual][API],

and other documented topics.

[full documentation]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/
[installation]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/#installation
[usage]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/#usage
[tutorials]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/Tutorials.html
[API]: https://mmg-gitlab.fjfi.cvut.cz/doc/tnl/annotated.html

## Getting involved

The TNL project welcomes and encourages participation by everyone. While most of the work for TNL
involves programming in principle, we value and encourage contributions even from people proficient
in other, non-technical areas.

This section provides several ideas how both new and experienced TNL users can contribute to the
project. Note that this is not an exhaustive list.

- Join the __code development__. Our [GitLab issues tracker][GitLab issues] contains many ideas for
  new features, or you may bring your own. The [contributing guidelines](CONTRIBUTING.md) describe
  the standards for code contributions.
- Help with __testing and reporting problems__. Testing is an integral part of agile software
  development which refines the code development. Constructive critique is always welcome.
- Improve and extend the __documentation__. Even small changes such as improving grammar or fixing
  typos are very appreciated.
- Share __your experience__ with TNL. Have you used TNL in your own project? Please be open and
  [share your experience][contact] to help others in similar fields to get familiar with TNL. If
  you could not utilize TNL as smoothly as possible, feel free to submit a [feature request][GitLab
  issues].

Before contributing, please get accustomed with the [code of conduct][code of conduct].

[GitLab issues]: https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/issues
[code of conduct]: CODE_OF_CONDUCT.md
[contact]: https://tnl-project.org/#contact

## Citing

If you use TNL in your scientific projects, please cite the following paper in
your publications:

- T. Oberhuber, J. Klinkovský, R. Fučík, [TNL: Numerical library for modern
  parallel architectures](https://ojs.cvut.cz/ojs/index.php/ap/article/view/6075),
  Acta Polytechnica 61.SI (2021), 122-134.

## Authors

See the [list of team members](https://tnl-project.org/about/) on our website.
The [overview of contributions](https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/-/graphs/develop)
can be viewed on GitLab.

## License

Template Numerical Library is provided under the terms of the [MIT License](
https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/blob/develop/LICENSE).
