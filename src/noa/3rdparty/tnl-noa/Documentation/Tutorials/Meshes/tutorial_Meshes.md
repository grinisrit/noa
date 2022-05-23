# Unstructured meshes tutorial

[TOC]

## Introduction

The [Mesh](@ref TNL::Meshes::Mesh) class template is a data structure for _conforming unstructured homogeneous_ meshes, which can be used as the fundamental data structure for numerical schemes based on finite volume or finite element methods.
The abstract representation supports almost any cell shape which can be described by an [entity topology](@ref TNL::Meshes::Topologies).
Currently there are common 2D quadrilateral, 3D hexahedron and arbitrarily dimensional simplex topologies built in the library.
The implementation is highly configurable via templates of the C++ language, which allows to avoid the storage of unnecessary dynamic data.
The internal memory layout is based on state--of--the--art [sparse matrix formats](@ref TNL::Matrices), which are optimized for different hardware architectures in order to provide high performance computations.
The [DistributedMesh](@ref TNL::Meshes::DistributedMeshes::DistributedMesh) class template is an extended data structure based on `Mesh`, which allows to represent meshes decomposed into several subdomains for distributed computing using the Message Passing Interface (MPI).

## Reading a mesh from a file

The most common way of mesh initialization is by reading a prepared input file created by an external program.
TNL provides classes and functions for reading the common VTK, VTU and Netgen file formats.

The main difficulty is mapping the mesh included in the file to the correct C++ type, which can represent the mesh stored in the file.
This can be done with the [MeshTypeResolver](@ref TNL::Meshes::MeshTypeResolver) class, which needs to be configured to enable the processing of the specific cell topologies, which we want our program to handle.
For example, in the following code we enable loading of 2D triangular and quadrangular meshes:

\snippet ReadMeshExample.cpp config

There are other build config tags which can be used to enable or disable specific types used in the mesh: `RealType`, `GlobalIndexType` and `LocalIndexType`.
See the [BuildConfigTags](@ref TNL::Meshes::BuildConfigTags) namespace for an overview of these tags.

Next, we can define the main task of our program as a templated function, which will be ultimately launched with the correct mesh type based on the input file.
We can also use any number of additional parameters, such as the input file name:

\snippet ReadMeshExample.cpp task

Of course in practice, the function would be much more complex than this example, where we just print the file name and some textual representation of the mesh to the standard output.

Finally, we define the `main` function, which sets the input parameters (hard-coded in this example) and calls the [resolveAndLoadMesh](@ref TNL::Meshes::resolveAndLoadMesh) function to resolve the mesh type and load the mesh from the file into the created object:

\snippet ReadMeshExample.cpp main

We need to specify two template parameters when calling `resolveAndLoadMesh`:

1. our build config tag (`MeshConfigTag` in this example),
2. and the [device](@ref TNL::Devices) where the mesh should be allocated.

Then we pass the the function which should be called with the initialized mesh, the input file name, and the input file format (`"auto"` means auto-detection based on the file name).
In order to show the flexibility of passing other parameters to our main `task` function as defined above, we suggest to implement a wrapper lambda function (called `wrapper` in the example), which captures the relevant variables and forwards them to the `task`.

The return value of the `resolveAndLoadMesh` function is a boolean value representing the success (`true`) or failure (`false`) of the whole function call chain.
Hence, the return type of both `wrapper` and `task` needs to be `bool` as well.

For completeness, the full example follows:
\includelineno ReadMeshExample.cpp

## Mesh configuration   {#configuration}

The [Mesh](@ref TNL::Meshes::Mesh) class template is configurable via its first template parameter, `Config`.
By default, the \ref TNL::Meshes::DefaultConfig template is used.
Alternative, user-specified configuration templates can be specified by defining the mesh configuration as the `MeshConfig` template in the [MeshConfigTemplateTag](@ref TNL::Meshes::BuildConfigTags::MeshConfigTemplateTag) build config tag specialization.
For example, here we derive the `MeshConfig` template from the [DefaultConfig](@ref TNL::Meshes::DefaultConfig) template and override the `subentityStorage` member function to store only those subentity incidence matrices, where the subentity dimension is 0 and the other dimension is at least $D-1$.
Hence, only faces and cells will be able to access their subvertices and there will be no other links from entities to their subentities.

\snippet MeshConfigurationExample.cpp Configuration example

## Public interface and basic usage

The whole public interface of the unstructured mesh and its mesh entity class can be found in the reference manual: \ref TNL::Meshes::Mesh, \ref TNL::Meshes::MeshEntity.
Here we describe only the basic member functions.

The main purpose of the [Mesh](@ref TNL::Meshes::Mesh) class template is to provide access to the mesh entities.
Firstly, there is a member function template called [getEntitiesCount](@ref TNL::Meshes::Mesh::getEntitiesCount) which returns the number of entities of the dimension specified as the template argument.
Given a mesh instance denoted as `mesh`, it can be used like this:

\snippet MeshIterationExample.cpp getEntitiesCount

Note that this member function and all other member functions presented below are marked as [\_\_cuda\_callable\_\_](../GeneralConcepts/tutorial_GeneralConcepts.md), so they can be called from usual host functions as well as CUDA kernels.

The entity of given dimension and index can be accessed via a member function template called [getEntity](@ref TNL::Meshes::Mesh::getEntity).
Again, the entity dimension is specified as a template argument and the index is specified as a method argument.
The `getEntity` member function does not provide a _reference_ access to an entity stored in the mesh, but each entity is created _on demand_ and contains only a pointer to the mesh and the supplied entity index.
Hence, the mesh entity is kind of a _proxy object_ where all member functions call just an appropriate member function via the mesh pointer.
The `getEntity` member function can be used like this:

\snippet MeshIterationExample.cpp getEntity

Here we assume that `idx < num_vertices` and `idx2 < num_cells`.
Note that both `Mesh::Vertex` and `Mesh::Cell` are specific instances of the [MeshEntity](@ref TNL::Meshes::MeshEntity) class template.

The information about the subentities and superentities can be accessed via the following member functions:

- [getSubentitiesCount](@ref TNL::Meshes::MeshEntity::getSubentitiesCount)
- [getSubentityIndex](@ref TNL::Meshes::MeshEntity::getSubentityIndex)
- [getSuperentitiesCount](@ref TNL::Meshes::MeshEntity::getSuperentitiesCount)
- [getSuperentityIndex](@ref TNL::Meshes::MeshEntity::getSuperentityIndex)

For example, they can be combined with the `getEntity` member function to iterate over all subvertices of a specific cell:

\snippet MeshIterationExample.cpp Iteration over subentities

The iteration over superentities adjacent to an entity is very similar and left as an exercise for the reader.

Note that the implementations of all templated member functions providing access to subentities and superentities contain a `static_assert` expression which checks if the requested subentities or superentities are stored in the mesh according to its [configuration](#configuration).

## Parallel iteration over mesh entities

The [Mesh](@ref TNL::Meshes::Mesh) class template provides a simple interface for the parallel iteration over mesh entities of a specific dimension.
There are several member functions:

- [forAll](@ref TNL::Meshes::Mesh::forAll) -- iterates over all mesh entities of a specific dimension
- [forBoundary](@ref TNL::Meshes::Mesh::forBoundary) -- iterates over boundary mesh entities of a specific dimension
- [forInterior](@ref TNL::Meshes::Mesh::forInterior) -- iterates over interior (i.e., not boundary) mesh entities of a specific dimension

For distributed meshes there are two additional member functions:

- [forGhost](@ref TNL::Meshes::Mesh::forGhost) -- iterates over ghost mesh entities of a specific dimension
- [forLocal](@ref TNL::Meshes::Mesh::forLocal) -- iterates over local (i.e., not ghost) mesh entities of a specific dimension

All of these member functions have the same interface: they take one parameter, which should be a functor, such as a _lambda expression_ $f$ that is called as $f(i)$, where $i$ is the mesh entity index in the current iteration.
Remember that the iteration is performed _in parallel_, so all calls to the functor must be independent since they can be executed in any order.

Note that only the mesh entity index is passed to the functor, it does not get the mesh entity object or even (a reference to) the mesh.
All additional information needed by the functor must be handled manually, e.g. via a _lambda capture_.

For example, the iteration over cells on a mesh allocated on the host can be done as follows:

\snippet MeshIterationExample.cpp Parallel iteration host

The parallel iteration is more complicated for meshes allocated on a GPU, since the lambda expression needs to capture a pointer to the copy of the mesh, which is allocated on the right device.
This can be achieved with a [smart pointer](../Pointers/tutorial_Pointers.md) as follows:

\snippet ParallelIterationCuda.h Parallel iteration CUDA

Alternatively, you can use a [SharedPointer](@ref TNL::Pointers::SharedPointer) instead of a [DevicePointer](@ref TNL::Pointers::DevicePointer) to allocate the mesh, but it does not allow to bind to an object which has already been created outside of the `SharedPointer`.

## Writing a mesh and data to a file

Numerical simulations typically produce results which can be interpreted as _mesh functions_ or _fields_.
In C++ they can be stored simply as arrays or vectors with the appropriate size.
For example, here we create two arrays `f_in` and `f_out`, which represent the input and output state of an iterative algorithm (`f_in` and `f_out` will be swapped after each iteration):

\snippet GameOfLife.cpp Data vectors

Note that here we used `std::uint8_t` as the value type.
The following value types are supported for the output into VTK file formats: `std::int8_t`, `std::uint8_t`, `std::int16_t`, `std::uint16_t`, `std::int32_t`, `std::uint32_t`, `std::int64_t`, `std::uint64_t`, `float`, `double`.

The output into a specific file format can be done with an appropriate _writer_ class, see \ref TNL::Meshes::Writers.
For example, using [VTUWriter](@ref TNL::Meshes::Writers::VTUWriter) for the `.vtu` file format:

\snippet GameOfLife.cpp make_snapshot

Note that this writer supports writing metadata (iteration index and time level value), then we call `writeEntities` to write the mesh cells and `writeCellData` to write the mesh function values.
The `writeCellData` call can be repeated multiple times for different mesh functions that should be included in the snapshot.

Then we can take the snapshot of the initial state,

\snippet GameOfLife.cpp make initial snapshot

and similarly use `make_snapshot` in the iteration loop.

## Example: Game of Life

In this example we will show how to implement the [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) using the [Mesh](@ref TNL::Meshes::Mesh) class template.
Although the game is usually implemented on structured grids rather than unstructured meshes, it will nicely illustrate how the building blocks for a numerical simulation are connected together.

The kernel of the Game of Life can be implemented as follows:

\snippet GameOfLife.cpp Game of Life kernel

The `kernel` function takes `f_in_view` (the input state of the current iteration) and for the $i$-th cell sums the values of the neighbor cells, which are accessed using the dual graph -- see \ref TNL::Meshes::Mesh::getCellNeighborsCount and \ref TNL::Meshes::Mesh::getCellNeighborIndex.
Then it writes the resulting state of the $i$-th cell into `f_out_view` according to Conway's rules for a square grid:

- any live cell with less than two live neighbors dies,
- any live cell with two or three live neighbors survives,
- any live cell with more than three live neighbors dies,
- any dead cell with exactly three live neighbors becomes a live cell,
- and any other dead cell remains dead.

The kernel function is evaluated for all cells in the mesh, followed by swapping `f_in` and `f_out` (including their views), writing the output into a VTU file and checking if this was the last iteration:

\snippet GameOfLife.cpp Game of Life iteration

The remaining pieces needed for the implementation have either been already presented on this page, or they are left as an exercise to the reader.
For the sake of completeness, we include the full example below.

\include GameOfLife.cpp
